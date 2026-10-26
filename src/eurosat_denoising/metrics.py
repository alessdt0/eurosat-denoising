import math
import torch

def squared_error_sum(pred, target):
    """Calcola la somma dei quadrati degli errori (SSE), senza normalizzarla."""
    # Somma, su tutto il batch, degli errori al quadrato tra predizione e bersaglio.
    return float((pred - target).square().sum().item())

def pixel_count(tensor):
    """Restituisce il numero di pixel (canali * altezza * larghezza) in un batch."""
    # Utile per normalizzare la SSE ottenendo un MSE globale.
    return tensor.numel()

def mse(pred, target):
    """Calcola il MSE globale (SSE divisa per il numero di pixel), non la media dei batch."""
    # Si normalizza per intero: evita distorsioni rispetto alla media dei MSE per batch.
    return squared_error_sum(pred, target) / pixel_count(pred)

def psnr(pred, target):
    """Calcola il PSNR per immagini normalizzate in ``[0, 1]``."""
    # PSNR = 10 * log10(1 / MSE); infinito se l'MSE e' nullo (ricostruzione perfetta).
    value = mse(pred, target)
    return float('inf') if value == 0 else 10 * math.log10(1 / value)

def ssim_scores(pred, target):
    """Restituisce un punteggio SSIM per ogni immagine RGB del batch."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError('Install scikit-image to compute SSIM') from exc
    # I tensori sono (batch, canali, altezza, larghezza): si traspongono per scikit-image.
    a, b = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    return [float(structural_similarity(x.transpose(1, 2, 0), y.transpose(1, 2, 0), channel_axis=2, data_range=1.0)) for x, y in zip(a, b)]
