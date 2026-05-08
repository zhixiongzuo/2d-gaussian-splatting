import torch
import torch.nn.functional as F

_window_cache = {}


def _create_gaussian_window(window_size, channel, device, dtype):
    key = (window_size, channel, device, dtype)
    cached = _window_cache.get(key)
    if cached is not None:
        return cached

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device="cpu")
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    _1d = g.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)

    _window_cache[key] = window
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    needs_squeeze = False
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        needs_squeeze = True

    channel = img1.size(1)
    window = _create_gaussian_window(window_size, channel, img1.device, img1.dtype)
    padding = window_size // 2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_stack = F.conv2d(
        torch.cat([img1, img2], dim=0),
        window,
        padding=padding,
        groups=channel,
    )
    mu1 = mu_stack[0:1]
    mu2 = mu_stack[1:2]

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    var_stack = F.conv2d(
        torch.cat([img1 * img1, img2 * img2, img1 * img2], dim=0),
        window,
        padding=padding,
        groups=channel,
    )
    sigma1_sq = var_stack[0:1] - mu1_sq
    sigma2_sq = var_stack[1:2] - mu2_sq
    sigma12 = var_stack[2:3] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3))
