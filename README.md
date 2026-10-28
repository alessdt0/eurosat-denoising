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

## Sweep sperimentale

For the AE vs VAE comparison across multiple degradation levels, use the `--sweep` mode:

```bash
python -m eurosat_denoising.cli --sweep --noise-stds 0.05,0.15,0.30 --blur-sigmas 1.0 --epochs 5 --models ae,vae
```

Available options: `--noise-stds` (comma-separated list of Gaussian noise levels, default `0.05,0.15,0.30`), `--blur-sigmas` (comma-separated list of Gaussian blur sigmas, default `1.0`; use an empty value like `--blur-sigmas ""` for no blur), `--models` (comma-separated list of architectures: `ae`, `vae`, `ae_skip`, `ae_skip32`, `ae_wide`, `ae_latent128`; default `ae,vae`), `--beta`, `--epochs`, `--batch-size`, `--overwrite`.

Every experiment is saved to `outputs/sweep/<arch>_noise_<sigma>_blur_<sigma>_epochs_<n>_beta_<b>_seed_<s>.json` with all experimental parameters and metrics (mse, psnr, ssim_mean, ssim_std, kl, val_mse, val_psnr, val_ssim_mean, val_ssim_std, n_params, baseline_mse, baseline_psnr). The filename embeds the experimental parameters, so different experiments never overwrite each other; if a JSON already exists the experiment is skipped (pass `--overwrite` to replace it). Every run is seeded independently, so different architectures see the same batch order and the same training noise.

## Ricerca della configurazione dell'autoencoder

La traccia chiede la *migliore configurazione dell'autoencoder*: le varianti architetturali si confrontano su una griglia ridotta e si sceglie il vincitore sul **validation set**, mai sul test.

```bash
python -m eurosat_denoising.cli --sweep --models ae,ae_skip,ae_skip32,ae_wide,ae_latent128 \
    --noise-stds 0.05,0.15 --blur-sigmas 0.5 --epochs 5 --sweep-dir outputs/sweep_arch
python -m eurosat_denoising.arch_report --sweep-dir outputs/sweep_arch --output-dir outputs/report_arch
```

| Architettura | Descrizione |
| --- | --- |
| `ae` | AutoEncoder di riferimento (2.17 M parametri, latente 64) |
| `ae_skip` | Skip a 16x16 che aggira il collo di bottiglia (2.24 M) |
| `ae_skip32` | Skip a 16x16 e a 32x32, tipo U-Net-lite (2.26 M) |
| `ae_wide` | Più canali nell'encoder (2.22 M) |
| `ae_latent128` | Latente 128 invece di 64 (4.26 M) |
| `vae` | VAE con latente 64 (3.22 M) |

`arch_report` stampa la tabella per architettura (parametri, PSNR di validation e di test, SSIM, guadagno sulla baseline), ordina per PSNR di validation e salva `arch_summary.json`, `arch_validation.txt`, `arch_psnr.png` e `arch_psnr_by_condition.png`.

## Report automatico

```bash
python -m eurosat_denoising.report --sweep-dir outputs/sweep --output-dir outputs/report
```

The report validates all sweep JSONs (required fields, filename/parameter coherence, metric sanity, baseline consistency, complete AE/VAE pairs), prints the experiment table and the AE vs VAE comparison, and saves `results.csv`, `summary.json`, `validation.txt` plus PNG plots (PSNR/MSE/SSIM bars, PSNR gain vs baseline, PSNR vs noise, AE vs VAE scatter). The notebook `notebooks/04_sweep_analysis.ipynb` uses the same functions for an interactive exploration.

## Presentazione

`presentation/slides.tex` is a 16:9 Beamer deck (16 slides) that compiles to `presentation/slides.pdf`; the spoken script is in `presentation/discorso.md`. Figures are read from `outputs/report/`.

```bash
cd presentation
pdflatex -interaction=nonstopmode slides.tex
pdflatex -interaction=nonstopmode slides.tex
```

Compile twice to update the total slide count in the footer. See `presentation/README.md` for requirements and customization.


## Notebooks

Run `01_explore_and_degrade.ipynb`, `02_autoencoder_training.ipynb`, `03_experiments_and_vae.ipynb`, and `04_sweep_analysis.ipynb` in order. They are exploratory views of the same task; the reusable implementation is in `src/eurosat_denoising` and the CLI is the canonical, non-notebook entrypoint.

## Experimental protocol

Use the same fixed 70/15/15 split, noise standard deviation, number of epochs, and evaluation metrics for AE and VAE. Report MSE, PSNR, SSIM when available, qualitative triplets (noisy/clean/reconstruction), parameter counts, and training time. The noisy-input baseline must also be reported: a model is useful only if it improves it.

The VAE includes a KL regularizer and may produce smoother reconstructions than the deterministic AutoEncoder. Results depend on CPU/GPU, random seed, beta, and training budget; do not claim one model is universally better from a single run.

## Protocollo scientifico

L'esperimento è un **conditional denoising VAE**: l'immagine degradata `y=D(x)` è l'ingresso e l'immagine pulita `x` è il target. L'AutoEncoder minimizza `MSE(x_hat,x)`. Il VAE minimizza `MSE(x_hat,x) + beta*KL(q(z|y)||N(0,I))`, con `z=mu+sigma*epsilon` durante il training e `z=mu` durante la valutazione per rendere i risultati deterministici.

Il confronto deve usare lo stesso split 70/15/15, seed, rumore di test, numero massimo di epoche, early stopping e criterio di checkpoint (validation reconstruction MSE). Il rumore gaussiano viene studiato separatamente con `sigma=0.05, 0.15, 0.30`; il blur è un esperimento distinto. Va sempre riportata la baseline dell'immagine rumorosa. MSE e PSNR sono calcolati sul test usando l'errore globale; SSIM è riportata come media e deviazione standard per immagine. Il latent space AE e VAE è impostato a 64 dimensioni per un confronto più sensato.

`beta=1e-4` è un punto di partenza, non una costante universale: per una relazione più completa si può ripetere il confronto con `beta=1e-3`. EuroSAT contiene immagini RGB 64x64, quindi l'implementazione corrente è intenzionalmente specializzata a questa risoluzione.
