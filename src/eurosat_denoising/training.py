from pathlib import Path
import torch
from torch import nn
from .metrics import mse, psnr, ssim

def run_epoch(model, loader, device, optimizer=None, vae=False, beta=1e-4):
    model.train(optimizer is not None); total=0; count=0
    for noisy,clean,_ in loader:
        noisy,clean=noisy.to(device),clean.to(device); out=model(noisy)
        if vae:
            reconstruction,mu,logvar=out; rec=nn.functional.mse_loss(reconstruction,clean); kl=-.5*torch.mean(1+logvar-mu.square()-logvar.exp()); loss=rec+beta*kl; prediction=reconstruction
        else: prediction=out; loss=nn.functional.mse_loss(prediction,clean)
        if optimizer: optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()*len(noisy); count += len(noisy)
    return total/count

def evaluate(model, loader, device, vae=False):
    model.eval(); ms=ps=ss=0.; count=0
    with torch.no_grad():
        for noisy,clean,_ in loader:
            out=model(noisy.to(device)); prediction=out[0] if vae else out; prediction=prediction.cpu(); k=len(noisy); count+=k
            ms+=mse(prediction,clean)*k; ps+=psnr(prediction,clean)*k
            try: ss+=ssim(prediction,clean)*k
            except RuntimeError: pass
    result={'mse':ms/count,'psnr':ps/count};
    if ss: result['ssim']=ss/count
    return result

def fit(model, train_loader, val_loader, device, epochs=5, lr=1e-3, path=None, vae=False):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr); history={'train':[],'validation':[]}; best=float('inf')
    for epoch in range(epochs):
        tr=run_epoch(model,train_loader,device,optimizer,vae); va=run_epoch(model,val_loader,device,vae=vae); history['train'].append(tr);history['validation'].append(va)
        if va<best: best=va; path and torch.save(model.state_dict(),path)
        print(f'Epoch {epoch+1}/{epochs}: train={tr:.6f} validation={va:.6f}')
    if path and Path(path).exists(): model.load_state_dict(torch.load(path,map_location=device))
    return history
