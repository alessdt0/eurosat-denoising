import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def seed_everything(seed=42):
 """Set the random seeds used by Python, NumPy and PyTorch."""
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def project_root(start=None):
 """Find the repository root starting from a path inside the project."""
 p=(start or Path.cwd()).resolve()
 for q in (p,*p.parents):
  if (q/'README.md').exists() and (q/'notebooks').exists(): return q
 raise FileNotFoundError('Project root not found')

def add_gaussian_noise(image,std=.15):
 """Add Gaussian noise to a tensor in ``[0, 1]`` and clip the result."""
 return (image+torch.randn_like(image)*std).clamp(0,1)


def add_blur(image,kernel_size=5,sigma=1.):
 """Apply Gaussian blur; this is kept separate from additive noise."""
 return transforms.GaussianBlur(kernel_size,sigma)(image)


class NoisyCleanDataset(Dataset):
 """Return a degraded image, its clean target and no classification label."""
 def __init__(self,base,indices,corruption,seed=None): self.base,self.indices,self.corruption,self.seed=base,list(indices),corruption,seed; self.to_tensor=transforms.ToTensor()
 def __len__(self): return len(self.indices)
 def __getitem__(self,i):
  image,_=self.base[self.indices[i]]; clean=image if torch.is_tensor(image) else self.to_tensor(image)
  if self.seed is None: noisy=self.corruption(clean)
  else:
   # Preserve the caller's RNG state so deterministic evaluation does
   # not change randomness used by the rest of the experiment.
   state=torch.random.get_rng_state(); torch.manual_seed(self.seed+i)
   try: noisy=self.corruption(clean)
   finally: torch.random.set_rng_state(state)
  return noisy,clean
def mse_score(pred,target):
    """Compute mean squared error for tensors with matching shapes."""
    return float(torch.mean((pred-target)**2))


def psnr_score(pred,target):
    """Compute PSNR assuming that the maximum image value is one."""
    mse=torch.mean((pred-target)**2).item(); return float('inf') if mse==0 else float(10*np.log10(1/mse))
