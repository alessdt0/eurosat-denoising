"""Esperimenti di denoising per EuroSAT."""
# Esporta i modelli di AutoEncoder e VAE usati nell'esperimento.
from .models import ConvAutoencoder, ConvVAE
# Esporta anche la factory che crea i dataset rumorosi/puliti.
from .data import make_datasets
# Elenco dei nomi pubblici del pacchetto, per rendere prevedibile l'import *.
__all__ = ["ConvAutoencoder", "ConvVAE", "make_datasets"]
