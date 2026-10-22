import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import make_datasets, SplitConfig
from .models import ConvAutoencoder, ConvVAE
from .training import fit, evaluate
from .utils import seed_everything

def main():
 """Run a complete experiment from the command line."""
 p=argparse.ArgumentParser(description='Train/evaluate EuroSAT denoising models')
 p.add_argument('--data',default='data'); p.add_argument('--output',default='outputs'); p.add_argument('--model',choices=['ae','vae'],default='ae'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--noise-std',type=float,default=.15); p.add_argument('--download',action='store_true'); args=p.parse_args()
 seed_everything(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base,(train,val,test)=make_datasets(args.data,args.noise_std,download=args.download)
 loaders=[DataLoader(x,args.batch_size,shuffle=i==0,num_workers=0) for i,x in enumerate((train,val,test))]; is_vae=args.model=='vae'; model=(ConvVAE() if is_vae else ConvAutoencoder()).to(device); out=Path(args.output);out.mkdir(exist_ok=True); checkpoint=out/f'{args.model}_best.pt'
 fit(model,loaders[0],loaders[1],device,args.epochs,path=checkpoint,vae=is_vae); result=evaluate(model,loaders[2],device,vae=is_vae); print(json.dumps(result,indent=2)); (out/f'{args.model}_metrics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
