# EuroSAT Denoising

Project work on restoring degraded EuroSAT RGB images with a convolutional AutoEncoder and a Variational AutoEncoder (VAE). The clean image is the target; the EuroSAT class label is retained only for optional analysis.

## Reproducible setup

```bash
conda env create -f environment.yml
conda activate eurosat-clean
python -m pip install -e .
python -m ipykernel install --user --name eurosat-clean --display-name eurosat-clean
```

The dataset is never committed. Download it and run a CPU-friendly baseline directly from the repository root:

```bash
python -m eurosat_denoising.cli --download --epochs 5 --model ae
python -m eurosat_denoising.cli --epochs 5 --model vae
pytest -q
```

Use `--epochs`, `--batch-size`, `--noise-std`, `--data`, and `--output` to reproduce or extend experiments. Results and checkpoints are written to `outputs/`, which is ignored by Git.

## Notebooks

Run `01_explore_and_degrade.ipynb`, `02_autoencoder_training.ipynb`, and `03_experiments_and_vae.ipynb` in order. They are exploratory views of the same task; the reusable implementation is in `src/eurosat_denoising` and the CLI is the canonical, non-notebook entrypoint.

## Experimental protocol

Use the same fixed 70/15/15 split, noise standard deviation, number of epochs, and evaluation metrics for AE and VAE. Report MSE, PSNR, SSIM when available, qualitative triplets (noisy/clean/reconstruction), parameter counts, and training time. The noisy-input baseline must also be reported: a model is useful only if it improves it.

The VAE includes a KL regularizer and may produce smoother reconstructions than the deterministic AutoEncoder. Results depend on CPU/GPU, random seed, beta, and training budget; do not claim one model is universally better from a single run.
