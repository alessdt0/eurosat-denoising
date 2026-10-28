import torch
from torch import nn
from .metrics import mse, psnr, ssim_scores

def run_epoch(model, loader, device, optimizer=None, vae=False, beta=1e-4):
    """Esegue una passata di training/valutazione e restituisce i componenti della loss.

    Per il VAE l'obiettivo e' la ricostruzione MSE + ``beta * KL``.
    """
    # Il training in realta' richiede un optimizer: se e' None si calcola solo la valutazione.
    model.train(optimizer is not None); total = rec_total = kl_total = count = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        # Con il VAE si passa sample=optimizer is not None per attivare il sampling stocastico in training.
        out = model(noisy, sample=optimizer is not None) if vae else model(noisy)
        if vae:
            prediction, mu, logvar = out
            # Loss di training: MSE di ricostruzione + beta * KL.
            rec = nn.functional.mse_loss(prediction, clean)
            kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
            loss = rec + beta * kl
        else:
            prediction, rec, kl = out, nn.functional.mse_loss(out, clean), torch.tensor(0.0, device=device)
            # Per l'AE la loss e' solo la MSE di ricostruzione (contribuzione KL nulla).
            loss = rec
        if optimizer:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Aggiorna i totali usando il numero di campioni effettivi nel batch.
        n = len(noisy); total += loss.item() * n; rec_total += rec.item() * n; kl_total += kl.item() * n; count += n
    # Restituisce loss media, media della ricostruzione e media del KL.
    return {'loss': total / count, 'reconstruction': rec_total / count, 'kl': kl_total / count}

def evaluate(model, loader, device, vae=False):
    """Valuta su un test set fisso MSE globale, PSNR, SSIM per immagine e KL per il VAE."""
    # Metodologia di valutazione: niente dropout/batchnorm, niente gradienti.
    model.eval(); sse = pixels = 0; ssims = []; kl_total = kl_count = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            # Con il VAE si usa sample=False (media del posterior) per metriche deterministiche.
            kl = 0.0
            if vae:
                prediction, mu, logvar = model(noisy, sample=False)
                # KL tra la posteriore q(z|y) e il prior normale N(0, I).
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_total += kl.item() * len(noisy); kl_count += len(noisy)
            else:
                prediction = model(noisy)
            prediction = prediction.cpu(); sse += (prediction - clean).square().sum().item(); pixels += prediction.numel(); ssims.extend(ssim_scores(prediction, clean))
    # MSE globale = SSE / numero di pixel; PSNR = 10 * log10(1 / MSE).
    value = sse / pixels
    return {'mse': value, 'psnr': float('inf') if value == 0 else 10 * torch.log10(torch.tensor(1 / value)).item(), 'ssim_mean': sum(ssims) / len(ssims), 'ssim_std': torch.tensor(ssims).std(unbiased=False).item(), 'kl': (kl_total / kl_count) if kl_count else 0.0}

def fit(model, train_loader, val_loader, device, epochs=5, lr=1e-3, path=None, vae=False, beta=1e-4, patience=3):
    """Addestra il modello con early stopping e checkpoint sul validation set."""
    # Ottimizzatore Adam per i parametri del modello.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); history = {'train': [], 'validation': [], 'reconstruction': [], 'kl': []}; best = float('inf'); stale = 0; best_state = None
    # Addestra per un numero massimo di epoche.
    for epoch in range(epochs):
        # Passata di training con calcolo dei gradienti e aggiornamento dei pesi.
        train = run_epoch(model, train_loader, device, optimizer, vae, beta); val = run_epoch(model, val_loader, device, vae=vae, beta=beta)
        # Salva le curve di training, validation e KL.
        history['train'].append(train['loss']); history['validation'].append(val['reconstruction']); history['reconstruction'].append(val['reconstruction']); history['kl'].append(val['kl'])
        # Se la ricostruzione sul validation migliora, salva il checkpoint del modello.
        if val['reconstruction'] < best:
            best, stale = val['reconstruction'], 0
            # Copia dei pesi migliori in memoria: senza path il modello
            # resterebbe altrimenti ai pesi dell'ultima epoca.
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            if path: torch.save(model.state_dict(), path)
        else: stale += 1
        print(f"Epoch {epoch+1}/{epochs}: train={train['loss']:.6f} val_rec={val['reconstruction']:.6f} val_kl={val['kl']:.6f}")
        # Early stopping: ferma l'addestramento se non ci sono miglioramenti per `patience` epoche.
        if stale >= patience: break
    # Ripristina i pesi migliori visti sul validation set.
    if best_state is not None: model.load_state_dict(best_state)
    return history
