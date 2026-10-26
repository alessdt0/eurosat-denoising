from __future__ import annotations
# Importa le classi e i moduli necessari per definire il dataset e le split.
from dataclasses import dataclass
from typing import Callable
from sklearn.model_selection import train_test_split
from torchvision.datasets import EuroSAT
from .utils import add_gaussian_noise, NoisyCleanDataset

@dataclass(frozen=True)
class SplitConfig:
    """Configurazione della suddivisione in train/validation/test."""
    # Frazione di training: 70% del dataset.
    train: float = .70
    # Frazione di validation: 15% (dopo aver rimosso il training).
    validation: float = .15
    # Seed per rendere la split stratificata riproducibile.
    seed: int = 42

class DenoisingDataset(NoisyCleanDataset):
    """Dataset rumoroso/pulito con etichetta della classe (ereditato da NoisyCleanDataset)."""
    def __init__(self, base, indices, corruption: Callable, seed: int|None=None):
        # Inizializza il dataset base ereditato (base, indici, corruzione, seed).
        super().__init__(base, list(indices), corruption, seed)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        # Recupera la coppia (noisy, clean) dal dataset base ereditato.
        noisy, clean = super().__getitem__(i)
        # Aggiunge l'etichetta della classe dell'immagine recuperata.
        image, label = self.base[self.indices[i]]
        # Restituisce la coppia (noisy, clean) con l'etichetta della classe.
        return noisy, clean, label

def make_datasets(root, noise_std=.15, split=SplitConfig(), download=False):
    """Carica EuroSAT e crea i dataset stratificati train/validation/test."""
    # Carica il dataset da root; se download e' True lo scarica se manca.
    base = EuroSAT(root=root, download=download)
    indices = list(range(len(base)))
    # I target sono gli indici delle classi, utilizzati per la stratifica'.
    labels = list(base.targets)
    # Primo split: allena sul train (split.train) e tiene da parte il resto.
    train_idx, remainder, train_labels, remainder_labels = train_test_split(
        indices, labels, test_size=1 - split.train, random_state=split.seed, stratify=labels
    )
    # Secondo split: separa validation (split.validation) da test.
    val_fraction = split.validation / (1 - split.train)
    val_idx, test_idx = train_test_split(
        remainder, test_size=1 - val_fraction, random_state=split.seed, stratify=remainder_labels
    )
    # Corruzione di default: rumore gaussiano con deviazione standard noise_std.
    corruption=lambda x: add_gaussian_noise(x, noise_std)
    # Training usa rumore casuale globale; validation e test usano rumore fissato.
    return base, (DenoisingDataset(base, train_idx, corruption), DenoisingDataset(base, val_idx, corruption, 1000), DenoisingDataset(base, test_idx, corruption, 2000))
