from pathlib import Path
import torch
from torch import nn
from .metrics import mse, psnr, ssim_scores

def run_epoch(model, loader, device, optimizer=None, vae=False, beta=1e-4):
    """Run one training/evaluation pass and return loss components.

    For the VAE, the objective is reconstruction MSE plus ``beta * KL``.
    """
    model.train(optimizer is not None); total = rec_total = kl_total = count = 0
    for noisy, clean, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy, sample=optimizer is not None) if vae else model(noisy)
        if vae:
            prediction, mu, logvar = out
            rec = nn.functional.mse_loss(prediction, clean)
            kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
            loss = rec + beta * kl
        else:
            prediction, rec, kl = out, nn.functional.mse_loss(out, clean), torch.tensor(0.0, device=device)
            loss = rec
        if optimizer:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        n = len(noisy); total += loss.item() * n; rec_total += rec.item() * n; kl_total += kl.item() * n; count += n
    return {'loss': total / count, 'reconstruction': rec_total / count, 'kl': kl_total / count}

def evaluate(model, loader, device, vae=False):
    """Evaluate global MSE/PSNR and per-image SSIM on a fixed test set."""
    model.eval(); sse = pixels = 0; ssims = []
    with torch.no_grad():
        for noisy, clean, _ in loader:
            out = model(noisy.to(device), sample=False) if vae else model(noisy.to(device))
            prediction = out[0] if vae else out
            prediction = prediction.cpu(); sse += (prediction - clean).square().sum().item(); pixels += prediction.numel(); ssims.extend(ssim_scores(prediction, clean))
    value = sse / pixels
    return {'mse': value, 'psnr': float('inf') if value == 0 else 10 * torch.log10(torch.tensor(1 / value)).item(), 'ssim_mean': sum(ssims) / len(ssims), 'ssim_std': torch.tensor(ssims).std(unbiased=False).item()}

def fit(model, train_loader, val_loader, device, epochs=5, lr=1e-3, path=None, vae=False, beta=1e-4, patience=3):
    """Train with validation checkpointing and early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); history = {'train': [], 'validation': [], 'reconstruction': [], 'kl': []}; best = float('inf'); stale = 0
    for epoch in range(epochs):
        train = run_epoch(model, train_loader, device, optimizer, vae, beta); val = run_epoch(model, val_loader, device, vae=vae, beta=beta)
        history['train'].append(train['loss']); history['validation'].append(val['reconstruction']); history['reconstruction'].append(val['reconstruction']); history['kl'].append(val['kl'])
        if val['reconstruction'] < best:
            best, stale = val['reconstruction'], 0
            if path: torch.save(model.state_dict(), path)
        else: stale += 1
        print(f"Epoch {epoch+1}/{epochs}: train={train['loss']:.6f} val_rec={val['reconstruction']:.6f} val_kl={val['kl']:.6f}")
        if stale >= patience: break
    if path and Path(path).exists(): model.load_state_dict(torch.load(path, map_location=device))
    return history
