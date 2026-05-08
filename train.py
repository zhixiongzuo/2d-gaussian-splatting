#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint, sample
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
    compute_vcd_score, compute_vcp_score,
    compute_edge_guided_densify_mask, compute_significance_prune_mask
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def parse_schedule(schedule_str):
    return [float(x) for x in schedule_str.split(",")]

def parse_iters(iters_str):
    return [int(x) for x in iters_str.split(",")]

def get_coarse_to_fine_scale(iteration, schedule, iters):
    scale = schedule[0]
    for i, thresh in enumerate(iters):
        if iteration >= thresh:
            scale = schedule[min(i + 1, len(schedule) - 1)]
    return scale

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    c2f_schedule = parse_schedule(opt.coarse_to_fine_schedule)
    c2f_iters = parse_iters(opt.coarse_to_fine_iters)
    resolution_scales = sorted(set(c2f_schedule), reverse=True)

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

    batch_size = opt.batch_size
    use_batch = batch_size > 1 and opt.batch_grad_accum
    vram_limit = opt.vram_limit
    adaptive_batch = use_batch and vram_limit < 1.0
    current_batch_size = batch_size

    if opt.coarse_to_fine:
        print(f"[Coarse-to-Fine] Schedule: {c2f_schedule}, Iters: {c2f_iters}")
    if opt.global_local:
        print(f"[Global-to-Local] Switch at iter {opt.global_local_switch_iter}")
    if opt.vcd:
        print(f"[VCD] K={opt.vcd_K}, error_thresh={opt.vcd_error_threshold}, score_thresh={opt.vcd_score_threshold}")
    if opt.vcp:
        print(f"[VCP] K={opt.vcp_K}, error_thresh={opt.vcp_error_threshold}, score_thresh={opt.vcp_score_threshold}")
    if opt.z_ordering:
        print(f"[Z-Ordering] Interval={opt.z_ordering_interval}")
    if opt.splat_bounding:
        print(f"[Splat Bounding] Ratio={opt.splat_bounding_ratio}")
    print(f"[Fast SSIM] {'ON' if opt.fast_ssim else 'OFF'}")
    if opt.scale_scheduler:
        print(f"[Scale Scheduler] start={opt.scale_scheduler_start}, cap={opt.scale_cap}")
    if opt.long_axis_split:
        print(f"[Long-Axis-Split] ON")
    if opt.edge_guided_densify:
        print(f"[Edge-Guided Densification] threshold_ratio={opt.edge_threshold_ratio}")
    if opt.significance_prune:
        print(f"[Significance Pruning] min_contribution={opt.min_contribution}")
    if use_batch:
        mode_str = "adaptive" if adaptive_batch else "fixed"
        print(f"[Batch Training] batch_size={batch_size}, mode={mode_str}, vram_limit={vram_limit:.0%}")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        if use_batch:
            if adaptive_batch and iteration % 100 == 0:
                vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if vram_used > vram_limit and current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size - 1)
                    if current_batch_size == 1:
                        use_batch = False
                    print(f"[Adaptive Batch] VRAM {vram_used:.1%} > {vram_limit:.0%}, reducing batch_size to {current_batch_size}")
                elif vram_used < vram_limit * 0.7 and current_batch_size < batch_size:
                    current_batch_size = min(batch_size, current_batch_size + 1)
                    print(f"[Adaptive Batch] VRAM {vram_used:.1%} < {vram_limit*0.7:.0%}, increasing batch_size to {current_batch_size}")

        if use_batch:
            n_sample = min(current_batch_size, len(viewpoint_stack))
            indices = sample(range(len(viewpoint_stack)), n_sample)
            batch_cams = [viewpoint_stack[i] for i in sorted(indices, reverse=True)]
            for i in sorted(indices, reverse=True):
                viewpoint_stack.pop(i)

            if opt.coarse_to_fine:
                current_scale = get_coarse_to_fine_scale(iteration, c2f_schedule, c2f_iters)
                closest_scale = min(resolution_scales, key=lambda s: abs(s - current_scale))
                if closest_scale in scene.train_cameras:
                    scaled_cams = scene.getTrainCameras(closest_scale)
                    if scaled_cams:
                        batch_cams = sample(scaled_cams, min(n_sample, len(scaled_cams)))

            batch_Ll1 = 0.0
            batch_loss = 0.0
            batch_dist_loss = 0.0
            batch_normal_loss = 0.0

            last_viewpoint_cam = None
            last_image = None
            last_radii = None
            last_visibility_filter = None

            for cam_idx, viewpoint_cam in enumerate(batch_cams):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_func(image, gt_image))

                lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
                lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

                rend_dist = render_pkg["rend_dist"]
                rend_normal = render_pkg['rend_normal']
                surf_normal = render_pkg['surf_normal']
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = lambda_normal * (normal_error).mean()
                dist_loss = lambda_dist * (rend_dist).mean()

                view_loss = (loss + dist_loss + normal_loss) / n_sample
                view_loss.backward()

                batch_Ll1 += Ll1.item()
                batch_loss += loss.item()
                batch_dist_loss += dist_loss.item()
                batch_normal_loss += normal_loss.item()

                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                last_viewpoint_cam = viewpoint_cam
                last_image = image.detach()
                last_radii = radii.detach()
                last_visibility_filter = visibility_filter.detach()

            batch_Ll1 /= n_sample
            batch_loss /= n_sample
            batch_dist_loss /= n_sample
            batch_normal_loss /= n_sample

            viewpoint_cam = last_viewpoint_cam
            image = last_image
            radii = last_radii
            visibility_filter = last_visibility_filter
            Ll1 = torch.tensor(batch_Ll1)
            loss = torch.tensor(batch_loss)
            dist_loss = torch.tensor(batch_dist_loss)
            normal_loss = torch.tensor(batch_normal_loss)

        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if opt.coarse_to_fine:
                current_scale = get_coarse_to_fine_scale(iteration, c2f_schedule, c2f_iters)
                closest_scale = min(resolution_scales, key=lambda s: abs(s - current_scale))
                if closest_scale in scene.train_cameras:
                    scaled_cams = scene.getTrainCameras(closest_scale)
                    if scaled_cams:
                        viewpoint_cam = scaled_cams[randint(0, len(scaled_cams)-1)]

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
                if use_batch:
                    loss_dict["Batch"] = str(batch_size)
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
                if not use_batch:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    use_vcd = opt.vcd
                    use_global_local = opt.global_local

                    vcd_score = None
                    if use_vcd:
                        vcd_score = compute_vcd_score(
                            gaussians.get_xyz, gaussians.get_opacity, radii, visibility_filter,
                            scene.getTrainCameras(), render, gaussians, pipe, background,
                            K=opt.vcd_K, error_threshold=opt.vcd_error_threshold
                        )

                    if use_vcd and vcd_score is not None:
                        gaussians.densify_and_prune_vcd(
                            opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                            20 if iteration > opt.opacity_reset_interval else None,
                            vcd_score=vcd_score, vcd_threshold=opt.vcd_score_threshold
                        )
                    elif use_global_local:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        phase = 'global' if iteration < opt.global_local_switch_iter else 'local'

                        edge_densify_mask = None
                        if opt.edge_guided_densify and iteration % 500 == 0 and iteration > 5000:
                            try:
                                edge_densify_mask, _ = compute_edge_guided_densify_mask(
                                    gaussians, image, grads, opt.densify_grad_threshold, viewpoint_cam,
                                    edge_threshold_ratio=opt.edge_threshold_ratio
                                )
                            except Exception:
                                edge_densify_mask = None

                        if opt.long_axis_split:
                            if edge_densify_mask is not None:
                                grads_masked = grads.clone()
                                grads_masked[~edge_densify_mask] = 0.0
                                gaussians.densify_and_split_long_axis(grads_masked, opt.densify_grad_threshold, scene.cameras_extent, N=2)
                            else:
                                gaussians.densify_and_split_long_axis(grads, opt.densify_grad_threshold, scene.cameras_extent, N=2)
                        else:
                            if edge_densify_mask is not None:
                                grads_masked = grads.clone()
                                grads_masked[~edge_densify_mask] = 0.0
                                gaussians.densify_and_split(grads_masked, opt.densify_grad_threshold, scene.cameras_extent, N=2)
                            else:
                                gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent, N=2)

                        if phase == 'local':
                            if edge_densify_mask is not None:
                                grads_clone = grads.clone()
                                grads_clone[~edge_densify_mask] = 0.0
                            else:
                                grads_clone = grads
                            gaussians.densify_and_clone(grads_clone, opt.densify_grad_threshold, scene.cameras_extent)

                        if opt.significance_prune:
                            prune_mask = compute_significance_prune_mask(
                                gaussians, gaussians.max_radii2D,
                                min_contribution=opt.min_contribution,
                                min_opacity=opt.opacity_cull
                            )
                        else:
                            prune_mask = (gaussians.get_opacity < opt.opacity_cull).squeeze()

                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        if size_threshold:
                            big_points_vs = gaussians.max_radii2D > size_threshold
                            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                    if opt.vcp and iteration > opt.global_local_switch_iter:
                        vcp_score = compute_vcp_score(
                            gaussians.get_xyz, gaussians.get_opacity, radii, visibility_filter,
                            scene.getTrainCameras(), render, gaussians, pipe, background,
                            K=opt.vcp_K, error_threshold=opt.vcp_error_threshold
                        )
                        gaussians.prune_vcp(
                            vcp_score, opt.vcp_score_threshold, opt.opacity_cull,
                            scene.cameras_extent,
                            20 if iteration > opt.opacity_reset_interval else None
                        )

                    if opt.splat_bounding:
                        gaussians.apply_splat_bounding()

                    if opt.scale_scheduler and iteration > opt.scale_scheduler_start:
                        gaussians.apply_scale_scheduler(
                            iteration,
                            scheduler_start=opt.scale_scheduler_start,
                            scale_cap=opt.scale_cap
                        )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

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
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid1())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
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
