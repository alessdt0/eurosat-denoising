import sys
from pathlib import Path
# Aggiunge 'src' al percorso di ricerca Python per importare il pacchetto senza installazione.
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
import torch
from eurosat_denoising.models import ARCHS, build_model, count_parameters, is_vae_arch

def test_all_architectures_reconstruct():
    """Ogni architettura deve restituire un'immagine 3x64x64 nel range [0, 1]."""
    # Batch casuale di immagini rumorose.
    x = torch.rand(2, 3, 64, 64)
    for name in ARCHS:
        model = build_model(name).eval()
        with torch.no_grad(): out = model(x)
        # Il VAE restituisce (ricostruzione, mu, logvar): si valuta solo la ricostruzione.
        reconstruction = out[0] if isinstance(out, tuple) else out
        assert reconstruction.shape == x.shape, name
        assert 0 <= reconstruction.min() and reconstruction.max() <= 1, name

def test_skip_connections_are_parameter_efficient():
    """Le skip connections non devono esplodere il numero di parametri."""
    # Il confronto e' con l'AE di riferimento: entro il 10% di parametri.
    reference = count_parameters(build_model('ae'))
    assert abs(count_parameters(build_model('ae_skip')) - reference) < 0.1 * reference

def test_vae_family_flag():
    """Solo il VAE deve attivare la loss con termine KL."""
    assert is_vae_arch('vae')
    assert not any(is_vae_arch(name) for name in ARCHS if name != 'vae')
