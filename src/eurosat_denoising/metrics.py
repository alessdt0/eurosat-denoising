import math
import torch

def squared_error_sum(pred, target):
    """Return the unnormalized squared error used for global aggregation."""
    return float((pred - target).square().sum().item())

def pixel_count(tensor):
    """Return the number of scalar pixels/channels in a batch tensor."""
    return tensor.numel()

def mse(pred, target):
    """Compute global MSE rather than averaging per-batch metric values."""
    return squared_error_sum(pred, target) / pixel_count(pred)

def psnr(pred, target):
    """Compute PSNR for images normalized to the range ``[0, 1]``."""
    value = mse(pred, target)
    return float('inf') if value == 0 else 10 * math.log10(1 / value)

def ssim_scores(pred, target):
    """Return one SSIM score per RGB image in a batch."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError('Install scikit-image to compute SSIM') from exc
    a, b = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    return [float(structural_similarity(x.transpose(1, 2, 0), y.transpose(1, 2, 0), channel_axis=2, data_range=1.0)) for x, y in zip(a, b)]
