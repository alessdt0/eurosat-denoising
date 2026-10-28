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

class SkipAutoencoder(nn.Module):
    """Autoencoder con skip connections (U-Net-lite) per il denoising.

    Le feature dell'encoder vengono riconcatenate al decoder, quindi i dettagli
    spaziali non devono essere compressi nel vettore latente. La skip a 16x16
    aggira il collo di bottiglia; quella opzionale a 32x32 recupera i bordi.
    """

    def __init__(self, channels=32, latent_dim=64, input_skip=True):
        """Costruisce encoder, collo di bottiglia e decoder con le skip."""
        super().__init__()
        # Flag che controlla la seconda skip a risoluzione 32x32.
        self.input_skip = input_skip
        # Encoder: 3x64x64 -> channels x 32x32 -> 64 x 16x16.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU(),
        )
        # Bottleneck latente invariato rispetto a ConvAutoencoder (64 dimensioni).
        self.to_latent = nn.Linear(64 * 16 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 16 * 16)
        # Skip principale: unisce le feature latenti (64) e quelle dell'encoder (64).
        self.bottleneck_fuse = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU())
        # Upsampling 16x16 -> 32x32.
        self.up = nn.ConvTranspose2d(64, channels, 4, 2, 1)
        # Skip opzionale a 32x32: canali del decoder + canali dell'encoder.
        self.mid_fuse = (
            nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.ReLU())
            if input_skip else nn.ReLU()
        )
        # Upsampling 32x32 -> 64x64 e uscita in [0, 1].
        self.out = nn.ConvTranspose2d(channels, 3, 4, 2, 1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        """Restituisce le due feature map intermedie usate dalle skip."""
        # Applica i due blocchi convoluzionali dell'encoder separatamente per
        # conservare la mappa a 32x32 necessaria alla skip.
        first = self.encoder[0:2](x)
        second = self.encoder[2:4](first)
        return first, second

    def forward(self, x):
        """Mappa un batch rumoroso ``(N, 3, 64, 64)`` nella ricostruzione pulita."""
        shallow, deep = self.encode(x)
        # Latente -> mappa 64x16x16, poi concatenazione con la feature profonda.
        latent = self.from_latent(self.to_latent(deep.flatten(1))).view(-1, 64, 16, 16)
        y = self.bottleneck_fuse(torch.cat([latent, deep], dim=1))
        y = self.up(y)
        # Seconda skip (se attiva) seguita dal blocco di fusione.
        y = self.mid_fuse(torch.cat([y, shallow], dim=1) if self.input_skip else y)
        return self.activation(self.out(y))

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


# Architetture disponibili: nome -> costruttore. `ae`/`vae` sono quelle dello
# sweep originale; le altre sono varianti per la ricerca della miglior
# configurazione dell'autoencoder (piu' o meno capaci, con o senza skip).
ARCHS = {
    'ae': lambda: ConvAutoencoder(channels=32, latent_dim=64),
    'ae_skip': lambda: SkipAutoencoder(channels=32, latent_dim=64, input_skip=False),
    'ae_skip32': lambda: SkipAutoencoder(channels=32, latent_dim=64, input_skip=True),
    'ae_wide': lambda: ConvAutoencoder(channels=64, latent_dim=64),
    'ae_latent128': lambda: ConvAutoencoder(channels=32, latent_dim=128),
    'vae': lambda: ConvVAE(latent_dim=64),
}
# Famiglia di appartenenza: serve per il termine KL e per il confronto AE vs VAE.
ARCH_FAMILY = {name: ('vae' if name == 'vae' else 'ae') for name in ARCHS}


def build_model(name):
    """Crea il modello richiesto (``ae``, ``vae`` o una variante architetturale)."""
    if name not in ARCHS:
        raise ValueError(f"architettura sconosciuta {name!r}; disponibili: {', '.join(ARCHS)}")
    return ARCHS[name]()


def is_vae_arch(name):
    """True se l'architettura e' un VAE (loss con termine KL)."""
    return ARCH_FAMILY[name] == 'vae'


def count_parameters(model):
    """Numero di parametri addestrabili del modello."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
