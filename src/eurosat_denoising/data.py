from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import EuroSAT
from .utils import add_gaussian_noise

@dataclass(frozen=True)
class SplitConfig:
    """Fractions and seed defining the reproducible stratified split."""
    train: float = .70
    validation: float = .15
    seed: int = 42

class DenoisingDataset(Dataset):
    """Wrap EuroSAT and expose ``(noisy, clean, class_id)`` samples."""
    def __init__(self, base, indices, corruption: Callable, seed: int|None=None):
        self.base, self.indices, self.corruption, self.seed = base, list(indices), corruption, seed
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        clean = image if torch.is_tensor(image) else self.to_tensor(image)
        if self.seed is None: noisy = self.corruption(clean)
        else:
            # Validation and test use fixed noise while leaving global RNG
            # state untouched for reproducible training.
            state = torch.random.get_rng_state(); torch.manual_seed(self.seed+i)
            try: noisy = self.corruption(clean)
            finally: torch.random.set_rng_state(state)
        return noisy, clean, label

def make_datasets(root, noise_std=.15, split=SplitConfig(), download=False):
    """Load EuroSAT and create stratified train/validation/test datasets."""
    base = EuroSAT(root=root, download=download)
    indices = list(range(len(base)))
    labels = list(base.targets)
    train_idx, remainder, train_labels, remainder_labels = train_test_split(
        indices, labels, test_size=1 - split.train, random_state=split.seed, stratify=labels
    )
    val_fraction = split.validation / (1 - split.train)
    val_idx, test_idx = train_test_split(
        remainder, test_size=1 - val_fraction, random_state=split.seed, stratify=remainder_labels
    )
    corruption=lambda x: add_gaussian_noise(x, noise_std)
    return base, (DenoisingDataset(base, train_idx, corruption), DenoisingDataset(base, val_idx, corruption, 1000), DenoisingDataset(base, test_idx, corruption, 2000))
