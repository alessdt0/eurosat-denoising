from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT
from .utils import add_gaussian_noise

@dataclass(frozen=True)
class SplitConfig:
    train: float = .70
    validation: float = .15
    seed: int = 42

class DenoisingDataset(Dataset):
    def __init__(self, base, indices, corruption: Callable, seed: int|None=None):
        self.base, self.indices, self.corruption, self.seed = base, list(indices), corruption, seed
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        clean = image if torch.is_tensor(image) else self.to_tensor(image)
        if self.seed is None: noisy = self.corruption(clean)
        else:
            state = torch.random.get_rng_state(); torch.manual_seed(self.seed+i)
            try: noisy = self.corruption(clean)
            finally: torch.random.set_rng_state(state)
        return noisy, clean, label

def make_datasets(root, noise_std=.15, split=SplitConfig(), download=False):
    base = EuroSAT(root=root, download=download)
    n=len(base); a=int(n*split.train); b=int(n*split.validation); sizes=[a,b,n-a-b]
    train, val, test = random_split(base, sizes, generator=torch.Generator().manual_seed(split.seed))
    corruption=lambda x: add_gaussian_noise(x, noise_std)
    return base, (DenoisingDataset(base, train.indices, corruption), DenoisingDataset(base, val.indices, corruption, 1000), DenoisingDataset(base, test.indices, corruption, 2000))
