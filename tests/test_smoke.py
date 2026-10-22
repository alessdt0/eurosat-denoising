import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
import torch
from eurosat_denoising.models import ConvAutoencoder, ConvVAE
from eurosat_denoising.utils import add_gaussian_noise

def test_noise_range_and_shape():
    """Noise preserves the batch shape and normalized image range."""
    x=torch.rand(2,3,64,64); y=add_gaussian_noise(x,.15)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1

def test_model_shapes():
    """Both models reconstruct the expected EuroSAT tensor shape."""
    x=torch.rand(2,3,64,64)
    assert ConvAutoencoder()(x).shape == x.shape
    assert ConvVAE()(x)[0].shape == x.shape
