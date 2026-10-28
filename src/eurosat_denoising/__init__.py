"""Esperimenti di denoising per EuroSAT."""
# Esporta i modelli di AutoEncoder e VAE usati nell'esperimento.
from .models import ARCHS, ConvAutoencoder, ConvVAE, SkipAutoencoder, build_model, count_parameters
# Esporta anche la factory che crea i dataset rumorosi/puliti.
from .data import make_datasets
# Esporta la configurazione e l'esecuzione del sweep sperimentale.
from .sweep import SweepConfig, run_sweep
# Elenco dei nomi pubblici del pacchetto, per rendere prevedibile l'import *.
__all__ = ["ConvAutoencoder", "ConvVAE", "SkipAutoencoder", "ARCHS", "build_model", "count_parameters", "make_datasets", "SweepConfig", "run_sweep"]
