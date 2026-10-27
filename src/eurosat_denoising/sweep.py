"""Sweep sperimentale per il confronto AutoEncoder vs VAE nel denoising EuroSAT."""
from __future__ import annotations
# Importa Path per costruire i percorsi dei file di output.
from dataclasses import dataclass
from pathlib import Path
# Importa json per serializzare i risultati del singolo esperimento.
import json
# Importa torch per il device e i calcoli tensor.
import torch
# Importa DataLoader per costruire i loader dei dataset.
from torch.utils.data import DataLoader
# Importa EuroSAT per ricostruire i dati grezzi.
from torchvision.datasets import EuroSAT
# Importa train_test_split per i split train/validation/test.
from sklearn.model_selection import train_test_split
# Importa la configurazione dei split.
from .data import SplitConfig
# Importa le metriche di denoising.
from .metrics import mse, psnr
# Importa i modelli di AutoEncoder e VAE.
from .models import ConvAutoencoder, ConvVAE
# Importa le funzioni di training e valutazione.
from .training import fit, evaluate
# Importa le funzioni di degradazione e dataset.
from .utils import add_blur, add_gaussian_noise, NoisyCleanDataset, seed_everything

# Valori di default del sweep.
SWEEP_MODELS: tuple = ("ae", "vae")
SWEEP_NOISE_STDS: tuple = (0.05, 0.15, 0.30)
SWEEP_BLUR_SIGMAS: tuple = (1.0,)
SWEEP_EPOCHS: int = 5
SWEEP_BETA: float = 1e-4
SWEEP_SEED: int = 42
SWEEP_BATCH_SIZE: int = 128
# Semi fissi per validation e test: stessi valori dei notebook e della CLI.
VAL_SEED: int = 1000
TEST_SEED: int = 2000


@dataclass(frozen=True)
class SweepConfig:
    """Configurazione del sweep: modelli, livelli di rumore, blur, training."""
    # Modelli da confrontare; default AE e VAE.
    models: tuple = SWEEP_MODELS
    # Livelli di rumore gaussiano sigma; default i tre livelli sperimentati.
    noise_stds: tuple = SWEEP_NOISE_STDS
    # Livelli di blur gaussiano; default una dilatazione media. None per nessun blur.
    blur_sigmas: tuple = SWEEP_BLUR_SIGMAS
    # Numero di epoche di training per ogni esperimento.
    epochs: int = SWEEP_EPOCHS
    # Coefficiente beta della regularizzazione KL del VAE.
    beta: float = SWEEP_BETA
    # Random seed per la riproducibilita.
    seed: int = SWEEP_SEED
    # Batch size usato per training e valutazione.
    batch_size: int = SWEEP_BATCH_SIZE
    # Se True il rumore viene applicato dopo il blur; altrimenti prima.
    blur_after_noise: bool = False


def make_corruption(noise_std, blur_sigma=None, kernel_size=5, blur_after_noise=False):
    """Crea la funzione di corruzione per il dataset rumoroso/pulito.

    Combina il rumore gaussiano (deviazione standard ``noise_std``) con un
    eventuale blur gaussiano (``blur_sigma``); l'ordine dei due degradi si
    sceglie con ``blur_after_noise``.
    """
    if blur_sigma is None:
        # Corruzione solo rumore gaussiano.
        return lambda x: add_gaussian_noise(x, noise_std)
    # Blur gaussiano con kernel e sigma fissati.
    blur = lambda x: add_blur(x, kernel_size, blur_sigma)
    if blur_after_noise:
        # Ordine: rumore poi blur.
        return lambda x: add_blur(add_gaussian_noise(x, noise_std), kernel_size, blur_sigma)
    # Ordine: blur poi rumore.
    return lambda x: add_gaussian_noise(blur(x), noise_std)


def _split_indices(dataset, split: SplitConfig = SplitConfig()):
    """Calcola gli indici train/validation/test con lo split stratificato."""
    indices = list(range(len(dataset)))
    labels = list(dataset.targets)
    train_idx, remainder, _, remainder_labels = train_test_split(indices, labels, test_size=1 - split.train, random_state=split.seed, stratify=labels)
    val_idx, test_idx = train_test_split(remainder, test_size=split.validation / (1 - split.train), random_state=split.seed, stratify=remainder_labels)
    return train_idx, val_idx, test_idx


def _baseline_scores(loader, device):
    """Calcola MSE e PSNR della baseline rumorosa (identita')."""
    sse = pixels = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            sse += (noisy - clean).square().sum().item(); pixels += noisy.numel()
    value = sse / pixels
    return value, (float('inf') if value == 0 else 10 * torch.log10(torch.tensor(1 / value)).item())


def experiment_filename(record):
    """Costruisce il nome univoco del file di output per ogni esperimento."""
    blur = record.get("blur_sigma")
    blur_part = "none" if blur is None else f"{blur:g}"
    return f"{record['model']}_noise_{record['noise_std']:g}_blur_{blur_part}_epochs_{record['epochs']}_beta_{record['beta']:g}_seed_{record['seed']}.json"


def run_sweep(config: SweepConfig, root, output_dir=None, download: bool = False):
    """Esegue il sweep di esperimenti AE vs VAE e salva i risultati senza sovrascrizioni.

    Per ogni combinazione (modello, livello di rumore, livello di blur) addestra,
    valuta sul test set e salva un file JSON in ``output_dir``; il nome del file
    contiene i parametri sperimentali, quindi gli esperimenti diversi non si
    sovrascrivono.
    """
    seed_everything(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = EuroSAT(root=root, download=download)
    train_idx, val_idx, test_idx = _split_indices(base, SplitConfig())
    out_dir = Path(output_dir) if output_dir is not None else Path('outputs') / 'sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in config.models:
        for noise_std in config.noise_stds:
            for blur_sigma in config.blur_sigmas:
                corruption = make_corruption(noise_std, blur_sigma, blur_after_noise=config.blur_after_noise)
                train = NoisyCleanDataset(base, list(train_idx), corruption)
                val = NoisyCleanDataset(base, list(val_idx), corruption, VAL_SEED)
                test = NoisyCleanDataset(base, list(test_idx), corruption, TEST_SEED)
                train_loader = DataLoader(train, config.batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val, config.batch_size, shuffle=False, num_workers=0)
                test_loader = DataLoader(test, config.batch_size, shuffle=False, num_workers=0)
                is_vae = model_name == 'vae'
                model = (ConvVAE() if is_vae else ConvAutoencoder()).to(device)
                fit(model, train_loader, val_loader, device, epochs=config.epochs, path=None, vae=is_vae, beta=config.beta)
                result = evaluate(model, test_loader, device, vae=is_vae)
                baseline_mse, baseline_psnr = _baseline_scores(test_loader, device)
                record = {
                    'model': model_name,
                    'noise_std': noise_std,
                    'blur_sigma': blur_sigma,
                    'epochs': config.epochs,
                    'beta': config.beta,
                    'seed': config.seed,
                    'batch_size': config.batch_size,
                    'mse': result['mse'],
                    'psnr': result['psnr'],
                    'ssim_mean': result['ssim_mean'],
                    'ssim_std': result['ssim_std'],
                    'kl': result.get('kl', 0.0),
                    'baseline_mse': baseline_mse,
                    'baseline_psnr': baseline_psnr,
                }
                path = out_dir / experiment_filename(record)
                path.write_text(json.dumps(record, indent=2))
                results.append(record)
                print(f"[sweep] salvato {path}")
    return results
