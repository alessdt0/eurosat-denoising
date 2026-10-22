import torch
from torch import nn

class ConvAutoencoder(nn.Module):
    """Deterministic denoiser with a 64-dimensional latent vector."""
    def __init__(self, channels=32, latent_dim=64):
        """Build an encoder-decoder specialized to EuroSAT's 64x64 images."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU(),
        )
        self.to_latent = nn.Linear(64 * 16 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, channels, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """Map a noisy batch ``(N, 3, 64, 64)`` to clean reconstructions."""
        h = self.encoder(x).flatten(1)
        return self.decoder(self.from_latent(self.to_latent(h)).view(-1, 64, 16, 16))

class ConvVAE(nn.Module):
    """Conditional denoising VAE; input is degraded, target is clean."""
    def __init__(self, latent_dim=64):
        """Build the conditional VAE encoder, latent distribution and decoder."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.logvar = nn.Linear(64 * 16 * 16, latent_dim)
        self.expand = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x, sample=True):
        """Reconstruct ``x`` and return reconstruction, mean and log variance.

        ``sample=False`` uses the posterior mean and is used for deterministic
        validation/test metrics; training keeps the stochastic reparameterized
        sample.
        """
        h = self.encoder(x).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        reconstruction = self.decoder(self.expand(z).view(-1, 64, 16, 16))
        return reconstruction, mu, logvar
