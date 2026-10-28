import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import make_datasets
from .models import ARCHS, build_model, count_parameters, is_vae_arch
from .sweep import SweepConfig, run_sweep
from .training import fit, evaluate
from .utils import seed_everything

def main():
    """Esegue un esperimento completo di denoising da riga di comando."""
# Il parser accetta: --data, --output, --model/--arch, --epochs, --batch-size,
# --noise-std, --download, --sweep, --noise-stds, --blur-sigmas, --models,
# --beta, --overwrite e --sweep-dir.
    p=argparse.ArgumentParser(description='Train/evaluate EuroSAT denoising models')
    p.add_argument('--data',default='data'); p.add_argument('--output',default='outputs')
    # Architettura del singolo esperimento: ae, vae o una variante (skip, wide, latent128).
    p.add_argument('--model','--arch',dest='model',choices=tuple(ARCHS),default='ae',
                   help='architettura: '+', '.join(ARCHS))
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--noise-std',type=float,default=.15); p.add_argument('--download',action='store_true'); p.add_argument('--sweep',action='store_true'); p.add_argument('--noise-stds',type=str,default='0.05,0.15,0.30'); p.add_argument('--blur-sigmas',type=str,default='1.0'); p.add_argument('--models',type=str,default='ae,vae'); p.add_argument('--beta',type=float,default=1e-4)
    # Con --overwrite si sovrascrivono i JSON gia' presenti (default: si saltano).
    p.add_argument('--overwrite',action='store_true')
    # Cartella dei JSON dello sweep: default <output>/sweep, sovrascrivibile.
    p.add_argument('--sweep-dir',default=None,help='cartella dei JSON (default: <output>/sweep)')
    args=p.parse_args()
    if args.sweep:
        # Modalita' sweep: itera su architetture e livelli di rumore/blur.
        noise_stds=tuple(float(s) for s in args.noise_stds.split(',') if s.strip()); models=tuple(m.strip() for m in args.models.split(',') if m.strip()); blur_sigmas=tuple(float(b) for b in args.blur_sigmas.split(',') if b.strip()) or (None,)
        unknown=[m for m in models if m not in ARCHS]
        if unknown: p.error(f"architetture sconosciute: {unknown}; disponibili: {', '.join(ARCHS)}")
        config=SweepConfig(models=models,noise_stds=noise_stds,blur_sigmas=blur_sigmas,epochs=args.epochs,beta=args.beta,overwrite=args.overwrite)
        results=run_sweep(config,root=Path(args.data),output_dir=Path(args.sweep_dir) if args.sweep_dir else Path(args.output)/'sweep',download=args.download)
        for r in results: print(json.dumps(r,indent=2))
        return
    seed_everything(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base,(train,val,test)=make_datasets(args.data,args.noise_std,download=args.download)
    # Crea i DataLoader per train, validation e test; solo il training viene mescolato.
    loaders=[DataLoader(x,args.batch_size,shuffle=i==0,num_workers=0) for i,x in enumerate((train,val,test))]; is_vae=is_vae_arch(args.model); model=build_model(args.model).to(device); out=Path(args.output);out.mkdir(exist_ok=True); checkpoint=out/f'{args.model}_best.pt'
    # Addestra il modello (con early stopping e checkpoint sul validation).
    fit(model,loaders[0],loaders[1],device,args.epochs,path=checkpoint,vae=is_vae); result=evaluate(model,loaders[2],device,vae=is_vae); result['arch']=args.model; result['n_params']=count_parameters(model); print(json.dumps(result,indent=2)); (out/f'{args.model}_metrics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
