import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .metrics import mse, psnr


def seed_everything(seed=42):
 """Imposta i semi casuali di Python, NumPy e PyTorch."""
 # Fissa il seed per i generatori casuali di ciascuna libreria.
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 # Ripete il seed anche sulle eventuali GPU.
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 # Disattiva il benchmark e rende deterministiche le operazioni di convoluzione.
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def project_root(start=None):
 """Cerca la radice del repository partendo da un percorso interno al progetto."""
 # Risale i genitori finche trova una cartella con README.md e la cartella notebooks.
 p=(start or Path.cwd()).resolve()
 for q in (p,*p.parents):
  if (q/'README.md').exists() and (q/'notebooks').exists(): return q
 raise FileNotFoundError('Project root not found')

def add_gaussian_noise(image,std=.15):
 """Aggiunge rumore gaussiano a un tensore in ``[0, 1]`` e applica il clipping."""
 # Rumore gaussiano (media 0, deviazione standard `std`) con clipping in [0, 1].
 return (image+torch.randn_like(image)*std).clamp(0,1)


def add_blur(image,kernel_size=5,sigma=1.):
 """Applica il filtro di blur gaussiano; e' una degradazione separata dal rumore."""
 return transforms.GaussianBlur(kernel_size,sigma)(image)


class NoisyCleanDataset(Dataset):
 """Restituisce un'immagine degradata, il suo bersaglio pulito e nessuna etichetta di classe."""
 def __init__(self,base,indices,corruption,seed=None): self.base,self.indices,self.corruption,self.seed=base,list(indices),corruption,seed; self.to_tensor=transforms.ToTensor()
 def __len__(self): return len(self.indices)
 def __getitem__(self,i):
  image,_=self.base[self.indices[i]]; clean=image if torch.is_tensor(image) else self.to_tensor(image)
  if self.seed is None: noisy=self.corruption(clean)
  else:
   # Validazione e test usano un rumore con seed fisso, preservando lo stato del generatore.
   state=torch.random.get_rng_state(); torch.manual_seed(self.seed+i)
   try: noisy=self.corruption(clean)
   finally: torch.random.set_rng_state(state)
  return noisy,clean
def mse_score(pred,target):
    """Delega all'implementazione del MSE globale di :mod:`eurosat_denoising.metrics`."""
    # Alias di compatibilita': l'implementazione del MSE globale vive in metrics.py.
    return mse(pred, target)


def psnr_score(pred,target):
    """Delega all'implementazione del PSNR di :mod:`eurosat_denoising.metrics`."""
    # Alias di compatibilita': delega al calcolo del PSNR basato sul MSE globale.
    return psnr(pred, target)
