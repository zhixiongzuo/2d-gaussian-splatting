import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

_window_cache = {}

def _get_window(window_size, channel, device):
    key = (window_size, channel, device)
    if key not in _window_cache:
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)])
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        _window_cache[key] = window
    return _window_cache[key]

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = _get_window(window_size, channel, img1.device)
    window = window.type_as(img1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    padding = window_size // 2
    stacked = torch.cat([img1, img2], dim=0)
    mu = F.conv2d(stacked, window.expand(2 * channel, 1, window_size, window_size), padding=padding, groups=2 * channel)
    mu1, mu2 = mu[:channel], mu[channel:]
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sq_stack = torch.cat([img1 * img1, img2 * img2, img1 * img2], dim=0)
    sq_conv = F.conv2d(sq_stack, window.expand(3 * channel, 1, window_size, window_size), padding=padding, groups=3 * channel)
    sigma1_sq = sq_conv[:channel] - mu1_sq
    sigma2_sq = sq_conv[channel:2 * channel] - mu2_sq
    sigma12 = sq_conv[2 * channel:] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
