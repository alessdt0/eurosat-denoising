import sys
from pathlib import Path
# Aggiunge alla lista dei percorsi di ricerca Python la cartella 'src' del progetto,
# cosi' i test possono importare il pacchetto eurosat_denoising senza installazione.
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
import torch
from eurosat_denoising.models import ConvAutoencoder, ConvVAE
from eurosat_denoising.utils import add_gaussian_noise

def test_noise_range_and_shape():
    """Il rumore generato deve mantenere forma e range dell'immagine."""
    # Crea un batch di immagini casuali in [0, 1].
    x=torch.rand(2,3,64,64); y=add_gaussian_noise(x,.15)
    # Verifica che forma e range dei valori restino validi dopo il rumore.
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1

def test_model_shapes():
    """Entrambi i modelli devono ricostruire l'immagine rumorosa di input."""
    # Prepara un batch di immagini casuali in [0, 1].
    x=torch.rand(2,3,64,64)
    # L'AutoEncoder deve restituire un output con la stessa shape del input.
    assert ConvAutoencoder()(x).shape == x.shape
    # Il VAE deve ricostruire anch'esso l'immagine rumorosa:
    # l'output e' la ricostruzione seguita da media e log-varianza.
    assert ConvVAE()(x)[0].shape == x.shape
