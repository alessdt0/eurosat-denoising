# EuroSAT Denoising

Studio della ricostruzione di immagini EuroSAT degradate mediante
convolutional AutoEncoder e Variational AutoEncoder.

## Obiettivi

- Analizzare il dataset EuroSAT.
- Applicare degradazioni controllate alle immagini.
- Addestrare un AutoEncoder per rimuovere il rumore.
- Confrontare diverse configurazioni del modello.
- Valutare un possibile confronto con un VAE.

## Notebook workflow

Run notebooks in order: `01_explore_and_degrade.ipynb`, `02_autoencoder_training.ipynb`, `03_experiments_and_vae.ipynb`. Create the environment with `conda env create -f environment.yml`, activate `conda activate eurosat-clean`, then register the Jupyter kernel with `python -m ipykernel install --user --name eurosat-clean --display-name eurosat-clean`. The dataset and generated checkpoints are ignored by Git.
