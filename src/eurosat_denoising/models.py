import torch
from torch import nn

class ConvAutoencoder(nn.Module):
    def __init__(self, channels=32, latent_channels=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,channels,3,2,1),nn.ReLU(),nn.Conv2d(channels,latent_channels,3,2,1),nn.ReLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(latent_channels,channels,4,2,1),nn.ReLU(),nn.ConvTranspose2d(channels,3,4,2,1),nn.Sigmoid())
    def forward(self,x): return self.decoder(self.encoder(x))

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__(); self.encoder=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU())
        self.mu=nn.Linear(64*16*16,latent_dim); self.logvar=nn.Linear(64*16*16,latent_dim); self.expand=nn.Linear(latent_dim,64*16*16)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
    def forward(self,x):
        h=self.encoder(x).flatten(1); mu,logvar=self.mu(h),self.logvar(h); z=mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        return self.decoder(self.expand(z).view(-1,64,16,16)),mu,logvar
