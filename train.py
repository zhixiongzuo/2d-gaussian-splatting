import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, fast_ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.accel_utils import (
    z_order_sort, splat_bounding, compute_vcd_score, compute_vcp_score,
    compute_edge_guided_densify_mask, compute_significance_prune_mask
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def parse_schedule(s):
    return [float(x) for x in s.split(",")]

def parse_iters(s):
    return [int(x) for x in s.split(",")]

def get_coarse_to_fine_scale(iteration, schedule, iters):
    for i in range(len(iters) - 1, -1, -1):
        if iteration >= iters[i]:
            return schedule[i]
    return schedule[0]

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    if opt.coarse_to_fine:
        resolution_scales = parse_schedule(opt.coarse_to_fine_schedule)
        coarse_to_fine_iters = parse_iters(opt.coarse_to_fine_iters)
    else:
        resolution_scales = [1.0]
        coarse_to_fine_iters = []

    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    ssim_func = fast_ssim if opt.fast_ssim else ssim

    if opt.fast_ssim:
        print("[Fast SSIM] ON")
    if opt.z_ordering:
        print(f"[Z-Ordering] Interval={opt.z_ordering_interval}")
    if opt.splat_bounding:
        print(f"[Splat Bounding] Ratio={opt.splat_bounding_ratio}")
    if opt.vcd:
        print(f"[VCD] K={opt.vcd_K}, error_thresh={opt.vcd_error_threshold}, score_thresh={opt.vcd_score_threshold}")
    if opt.vcp:
        print(f"[VCP] K={opt.vcp_K}, error_thresh={opt.vcp_error_threshold}, score_thresh={opt.vcp_score_threshold}")
    if opt.global_local:
        print(f"[Global-to-Local] Switch at iter {opt.global_local_switch_iter}")
    if opt.coarse_to_fine:
        print(f"[Coarse-to-Fine] Schedule: {resolution_scales}, Iters: {coarse_to_fine_iters}")
    if opt.scale_scheduler:
        print(f"[Scale Scheduler] start={opt.scale_scheduler_start}, cap={opt.scale_cap}")
    if opt.long_axis_split:
        print("[Long-Axis-Split] ON")
    if opt.edge_guided_densify:
        print(f"[Edge-Guided Densification] threshold_ratio={opt.edge_threshold_ratio}")
    if opt.significance_prune:
        print(f"[Significance Pruning] min_contribution={opt.min_contribution}, min_opacity={opt.min_opacity_prune}")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if opt.coarse_to_fine:
            current_scale = get_coarse_to_fine_scale(iteration, resolution_scales, coarse_to_fine_iters)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras(current_scale).copy()
        else:
            current_scale = 1.0

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(current_scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_func(image, gt_image))
        
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    if opt.vcd:
                        vcd_score = compute_vcd_score(
                            gaussians.get_xyz, gaussians.get_opacity, radii, visibility_filter,
                            scene.getTrainCameras(current_scale), render, gaussians, pipe, background,
                            K=opt.vcd_K, error_threshold=opt.vcd_error_threshold
                        )
                        gaussians.densify_and_prune_vcd(
                            opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                            size_threshold, vcd_score, opt.vcd_score_threshold
                        )
                    elif opt.global_local:
                        if iteration < opt.global_local_switch_iter:
                            gaussians.densify_and_split(
                                gaussians.xyz_gradient_accum / gaussians.denom,
                                opt.densify_grad_threshold, scene.cameras_extent
                            )
                            prune_mask = (gaussians.get_opacity < opt.opacity_cull).squeeze()
                            if size_threshold is not None:
                                big_points_vs = gaussians.max_radii2D > size_threshold
                                big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                            gaussians.prune_points(prune_mask)
                        else:
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                    if opt.edge_guided_densify and iteration > 5000:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        edge_mask, _ = compute_edge_guided_densify_mask(
                            gaussians, image.detach(), grads, opt.densify_grad_threshold,
                            viewpoint_cam, opt.edge_threshold_ratio
                        )
                        if edge_mask.any():
                            gaussians.densify_and_clone_edge(grads, opt.densify_grad_threshold, scene.cameras_extent, edge_mask)

                    if opt.long_axis_split:
                        gaussians.densify_and_split_long_axis(opt.densify_grad_threshold, scene.cameras_extent)

                    if opt.significance_prune and iteration > 3000:
                        sig_mask = compute_significance_prune_mask(
                            gaussians, radii, opt.min_contribution, opt.min_opacity_prune
                        )
                        if sig_mask.any():
                            gaussians.prune_points(sig_mask)

                    if opt.splat_bounding:
                        bound_mask = splat_bounding(gaussians.get_scaling, opt.splat_bounding_ratio)
                        prune_mask = ~bound_mask
                        if prune_mask.any():
                            gaussians.prune_points(prune_mask)

                if opt.vcp and iteration > opt.global_local_switch_iter and iteration % opt.densification_interval == 0:
                    vcp_score = compute_vcp_score(
                        gaussians.get_xyz, gaussians.get_opacity, radii, visibility_filter,
                        scene.getTrainCameras(current_scale), render, gaussians, pipe, background,
                        K=opt.vcp_K, error_threshold=opt.vcp_error_threshold
                    )
                    vcp_mask = vcp_score < opt.vcp_score_threshold
                    if vcp_mask.any():
                        gaussians.prune_points(vcp_mask)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.scale_scheduler and iteration > opt.scale_scheduler_start:
                gaussians.apply_scale_scheduler(opt.scale_cap)

            if opt.z_ordering and iteration % opt.z_ordering_interval == 0:
                gaussians.apply_z_ordering()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {"#": gaussians.get_opacity.shape[0], "loss": ema_loss_for_log}
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') if os.getenv('OAR_JOB_ID') else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
    print("\nTraining complete.")
