import math
import torch

def mse(pred,target): return float(torch.mean((pred-target)**2).item())
def psnr(pred,target):
    value=mse(pred,target); return float('inf') if value==0 else 10*math.log10(1/value)
def ssim(pred,target):
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc: raise RuntimeError('Install scikit-image to compute SSIM') from exc
    a=pred.detach().cpu().numpy(); b=target.detach().cpu().numpy(); scores=[]
    for x,y in zip(a,b): scores.append(structural_similarity(x.transpose(1,2,0),y.transpose(1,2,0),channel_axis=2,data_range=1.0))
    return float(sum(scores)/len(scores))
