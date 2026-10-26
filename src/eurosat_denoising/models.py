import torch
from torch import nn

class ConvAutoencoder(nn.Module):
    """Autoencoder deterministico (denoiser) con vettore latente di 64 dimensioni."""
    def __init__(self, channels=32, latent_dim=64):
        """Build an encoder-decoder specialized to EuroSAT's 64x64 images."""
        # Inizializza il modulo e tutti i sotto-modelli neurali.
        super().__init__()
        # Encoder: riduce la risoluzione (stride 2) fino a un feature map 64x16x16.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU(),
        )
        # Lo spazio latente ha `latent_dim` dimensioni (default: 64).
        self.to_latent = nn.Linear(64 * 16 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 16 * 16)
        # Decoder: riemerge la risoluzione e produce un'immagine in [0, 1] con Sigmoid.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, channels, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """Mappa un batch rumoroso ``(N, 3, 64, 64)`` nella ricostruzione pulita."""
        # Passo completo: encoder -> appiattimento dei feature map -> bottleneck
        # latente -> decoder.
        h = self.encoder(x).flatten(1)
        return self.decoder(self.from_latent(self.to_latent(h)).view(-1, 64, 16, 16))

class ConvVAE(nn.Module):
    """VAE condizionato al denoising: l'ingresso e' rumoroso, il bersaglio pulito."""
    def __init__(self, latent_dim=64):
        """Build the conditional VAE encoder, latent distribution and decoder."""
        # Inizializza il modulo e tutti i sotto-modelli neurali.
        super().__init__()
        # Encoder: riduce la risoluzione fino a un feature map 64x16x16.
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        # Parametrizzazione della distribuzione latente (media e log-varianza).
        self.mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.logvar = nn.Linear(64 * 16 * 16, latent_dim)
        self.expand = nn.Linear(latent_dim, 64 * 16 * 16)
        # Decoder: riemerge la risoluzione e ricostruisce l'immagine in [0, 1].
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x, sample=True):
        """Ricostruisce ``x`` restituendo (ricostruzione, media, log-varianza).

        Con ``sample=False`` si usa la media del posterior, per metriche
        deterministiche in validazione e test; durante l'addestramento si usa il
        campione stocastico (reparameterization trick).
        """
        # Passo: encoder -> media e log-varianza della latente -> decoder.
        h = self.encoder(x).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        # z = mu + sigma * epsilon con sigma = exp(0.5 * logvar);
        # se sample e' False si usa la media mu come punto di fissaggio.
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        reconstruction = self.decoder(self.expand(z).view(-1, 64, 16, 16))
        return reconstruction, mu, logvar
