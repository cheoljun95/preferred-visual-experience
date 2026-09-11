import numpy as np
import torch
import yaml
from pathlib import Path
import sys
sys.path.append('..')
from model.drivingscene_vae import DrivingSceneVAE

COLOR_MEAN = [0.5, 0.5, 0.5]
COLOR_STD = [0.5, 0.5, 0.5]

def inverse_transform(arr):
    '''
    Inverse torch array (C*H*W) to (H*W*C) 1~255 unit8 numpy array
    '''
    
    arr = arr.cpu().detach().numpy()
    #mean = [0.28689554, 0.32513303, 0.28389177]
    #std = [0.18696375, 0.19017339, 0.18720214]
    mean=COLOR_MEAN
    std=COLOR_STD
    mean = np.array(mean)
    std = np.array(std)
    arr = arr.transpose(1, 2, 0)
    arr = (arr * std) + mean
    arr = np.clip(arr,0,1)
    arr = (arr * 255).astype(np.uint8)

    return arr

def load_model(version_dir, MODEL, mode='best',verbose=False):
    version_dir = Path(version_dir)
    # get configuration information
    try:
        cfg = yaml.load(open(version_dir / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open(version_dir.parent / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    version_name = [f for f in (version_dir / 'lightning_logs').glob('version_*')][0].stem
    if mode == 'best':
        checkpoint_path = [f for f in Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')
                           if 'best' in f.name and not 'bestaln' in f.name]
        checkpoint_path = checkpoint_path[-1]
    else:
        checkpoint_path = [f for f in
                           Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')]

        def get_epoch(fileName):
            epoch = [n for n in fileName.split('-') if 'epoch' in n][0]
            return int(epoch.split('=')[-1])

        checkpoint_path.sort(key=lambda f: get_epoch(f.name))
        checkpoint_path = checkpoint_path[-1]
        
    if verbose:
        print(f'Model is loaded from {str(checkpoint_path)}')
    model = MODEL(**cfg['model_configs'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model = model.eval()

    return model, cfg


class Decoder():
    def __init__(self, version_dir, center, device='cpu'):
        model, cfg = load_model(version_dir,DrivingSceneVAE,mode='latest')
        self.decoder = model.net.decoder.eval().to(device)
        self.center = np.load(center)
        self.device = device
    
    def __call__(self, z, scale, do_scale=True):
        if do_scale:
            z_normalized =z/np.linalg.norm(z)
            latent = z_normalized*scale+self.center
        else:
            latent = z+self.center
        with torch.no_grad():
            latent = torch.from_numpy(latent).to(self.device).float()[None,:,None,None,None]
            decoded = self.decoder(latent)
            decoded= decoded[0].transpose(0,1)

        images = [inverse_transform(d) for d in decoded]
        return images