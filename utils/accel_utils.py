import math
import torch
import torch.nn.functional as F

def compute_morton_code(x, y, z):
    N = x.shape[0]
    morton = torch.zeros(N, dtype=torch.int64, device=x.device)
    for i in range(21):
        morton |= ((x >> i) & 1) << (3 * i)
        morton |= ((y >> i) & 1) << (3 * i + 1)
        morton |= ((z >> i) & 1) << (3 * i + 2)
    return morton

def z_order_sort(gaussians_xyz):
    xyz_min = gaussians_xyz.min(dim=0).values
    xyz_max = gaussians_xyz.max(dim=0).values
    xyz_range = xyz_max - xyz_min
    xyz_range = torch.where(xyz_range < 1e-8, torch.ones_like(xyz_range), xyz_range)
    normalized = (gaussians_xyz - xyz_min) / xyz_range
    quantized = (normalized * ((1 << 21) - 1)).clamp(0, (1 << 21) - 1).to(torch.int64)
    morton_codes = compute_morton_code(quantized[:, 0], quantized[:, 1], quantized[:, 2])
    indices = torch.argsort(morton_codes)
    return indices

def splat_bounding(scaling, max_radius_ratio=10.0):
    max_scale = scaling.max(dim=1).values
    median_scale = torch.median(max_scale)
    mask = max_scale <= max_radius_ratio * median_scale
    return mask

@torch.no_grad()
def compute_vcd_score(gaussians_xyz, gaussians_opacity, radii, visibility_filter,
                      train_cameras, render_func, gaussians_model, pipe, background,
                      K=3, error_threshold=0.2):
    N = gaussians_xyz.shape[0]
    score = torch.zeros(N, device=gaussians_xyz.device)
    view_count = torch.zeros(N, device=gaussians_xyz.device)
    num_cameras = len(train_cameras)
    perm = torch.randperm(num_cameras)[:K]
    for ci in perm:
        camera = train_cameras[ci]
        render_pkg = render_func(camera, gaussians_model, pipe, background)
        rendered = render_pkg["render"]
        gt_image = camera.original_image.cuda()
        error_map = (rendered - gt_image).abs().mean(dim=0)
        high_error = (error_map > error_threshold).float().unsqueeze(0).unsqueeze(0)
        kernel_size = 7
        padding = kernel_size // 2
        high_error_density = F.avg_pool2d(high_error, kernel_size=kernel_size, stride=1, padding=padding).squeeze()
        viewspace_points = render_pkg["viewspace_points"]
        cam_radii = render_pkg["radii"].float()
        cam_visibility = render_pkg["visibility_filter"]
        H, W = error_map.shape
        u = viewspace_points[:, 0].long().clamp(0, W - 1)
        v = viewspace_points[:, 1].long().clamp(0, H - 1)
        local_density = high_error_density[v, u]
        area = math.pi * cam_radii ** 2
        vis = cam_visibility
        score[vis] += local_density[vis] * area[vis]
        view_count[vis] += 1
    valid = view_count > 0
    score[valid] /= view_count[valid]
    return score

@torch.no_grad()
def compute_vcp_score(gaussians_xyz, gaussians_opacity, radii, visibility_filter,
                      train_cameras, render_func, gaussians_model, pipe, background,
                      K=3, error_threshold=0.2):
    N = gaussians_xyz.shape[0]
    score = torch.zeros(N, device=gaussians_xyz.device)
    view_count = torch.zeros(N, device=gaussians_xyz.device)
    num_cameras = len(train_cameras)
    perm = torch.randperm(num_cameras)[:K]
    for ci in perm:
        camera = train_cameras[ci]
        render_pkg = render_func(camera, gaussians_model, pipe, background)
        rendered = render_pkg["render"]
        gt_image = camera.original_image.cuda()
        error_map = (rendered - gt_image).abs().mean(dim=0)
        high_error = (error_map > error_threshold).float().unsqueeze(0).unsqueeze(0)
        kernel_size = 7
        padding = kernel_size // 2
        high_error_density = F.avg_pool2d(high_error, kernel_size=kernel_size, stride=1, padding=padding).squeeze()
        viewspace_points = render_pkg["viewspace_points"]
        cam_radii = render_pkg["radii"].float()
        cam_visibility = render_pkg["visibility_filter"]
        H, W = error_map.shape
        u = viewspace_points[:, 0].long().clamp(0, W - 1)
        v = viewspace_points[:, 1].long().clamp(0, H - 1)
        local_density = high_error_density[v, u]
        area = math.pi * cam_radii ** 2
        opacity = gaussians_opacity.squeeze()
        vis = cam_visibility
        score[vis] += local_density[vis] * area[vis] * (1 - opacity[vis])
        view_count[vis] += 1
    valid = view_count > 0
    score[valid] /= view_count[valid]
    return score

@torch.no_grad()
def compute_laplacian_edge_map(rendered_image, median_normalize=True, percentile=95):
    gray = rendered_image.mean(dim=0)
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=rendered_image.device)
    laplacian = F.conv2d(gray.unsqueeze(0).unsqueeze(0), laplacian_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    edge_strength = torch.abs(laplacian)
    if median_normalize:
        median = torch.median(edge_strength)
        max_val = torch.quantile(edge_strength, percentile / 100.0)
        edge_strength = torch.clamp((edge_strength - median) / (max_val - median + 1e-8), 0, 1)
    edge_strength = F.avg_pool2d(edge_strength.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0).squeeze(0)
    return edge_strength

@torch.no_grad()
def compute_edge_guided_densify_mask(gaussians, rendered_image, grads, grad_threshold, viewpoint_cam, edge_threshold_ratio=0.3):
    edge_map = compute_laplacian_edge_map(rendered_image)
    grad_magnitude = torch.norm(grads, dim=-1) if grads.dim() > 1 else grads.squeeze()
    projected_2d = viewpoint_cam.project_to_image(gaussians.get_xyz)
    H, W = edge_map.shape
    u = projected_2d[:, 0].long().clamp(0, W - 1)
    v = projected_2d[:, 1].long().clamp(0, H - 1)
    edge_scores = edge_map[v, u]
    edge_threshold = edge_scores.mean() * edge_threshold_ratio
    grad_high = grad_magnitude >= grad_threshold
    edge_high = edge_scores > edge_threshold
    densify_mask = torch.logical_and(grad_high, edge_high)
    return densify_mask, edge_scores

def compute_significance_prune_mask(gaussians, radii, min_contribution=0.001, min_opacity=0.1):
    area = math.pi * radii ** 2
    opacity = gaussians.get_opacity.squeeze()
    contribution = area * opacity
    prune_mask = torch.logical_and(contribution < min_contribution, opacity < min_opacity)
    return prune_mask

@torch.no_grad()
def get_long_axis_direction(rotations):
    quat = rotations
    x = 2 * (quat[:, 1] * quat[:, 3] + quat[:, 0] * quat[:, 2])
    y = 2 * (quat[:, 2] * quat[:, 3] - quat[:, 0] * quat[:, 1])
    z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    long_axis_3d = torch.stack([x, y, z], dim=1)
    long_axis_3d = long_axis_3d / (torch.norm(long_axis_3d, dim=1, keepdim=True) + 1e-8)
    return long_axis_3d
