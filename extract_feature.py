from pathlib import Path
import tqdm
import numpy as np
import tqdm
from dataset.datamodule import DrivingSeqDataModule
from model.drivingscene_vae import DrivingSceneVAE
import hydra
import torch
from utils.load_utils import load_cfg_and_ckpt_path
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--output_path",type=str)
parser.add_argument("--save_path",type=str, )
parser.add_argument("--mode",type=str, default="latest")
parser.add_argument("--device",type=str, default="cuda:0")
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--save_std",default=False, action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    version_dir = args.output_path
    mode = args.mode
    device = args.device
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    cfg, ckpt_path = load_cfg_and_ckpt_path(version_dir,mode=mode)
    exp_name = cfg['experiment_name'] if args.exp_name is None else args.exp_name
    save_path = save_path/exp_name
    (save_path).mkdir(exist_ok=True)
    
    metainfo = {'exp_name':exp_name,
                'version_dir':version_dir,
                'mode':mode,
                'ckpt_path':ckpt_path,
                'config':cfg,}
                
    with open(save_path/"config.json", "w") as outfile:
        json.dump(metainfo, outfile)
        
    model = DrivingSceneVAE(**cfg['model_configs']) 
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=False)
    torch.save( {'config':cfg,'state_dict':ckpt['state_dict']},save_path/'model.ckpt')
    model = model.eval().to(device)
    datamodule = DrivingSeqDataModule(resize=[cfg['model_configs']['vae_configs']['input_shape'][1],
                                              cfg['model_configs']['vae_configs']['input_shape'][2]], **cfg['data_configs'])
    
    dataloader = datamodule.all_dataloader()
    frame_ids = dataloader.dataset.frame_ids
    
    avg_latent_feature = 0
    cnt = 0
    for batch in tqdm.tqdm(dataloader):
        imgs, indices = batch['images'],batch['index']
        imgs = torch.transpose(imgs, 1, 2) # so imgs: (batch,channel,seq,...)
        imgs = imgs.float() 
        imgs = imgs.to(device)
        with torch.no_grad():
            latent_features,var = model._get_latent_feats(imgs, return_var=True)
        latent_features = latent_features.detach().cpu().numpy()
        var = var.detach().cpu().numpy()
        avg_latent_feature = (avg_latent_feature*cnt + latent_features.sum(0))/(cnt+len(latent_features))
        cnt += len(latent_features)
        for i, latent_feature, v  in zip(indices,latent_features, var):
            frame_id = frame_ids[i]
            run_id,frame_num = frame_id.split('_')
            (save_path/run_id).mkdir(exist_ok=True)
            if args.save_std:
                ft = np.stack([latent_feature, v])
            else:
                ft = latent_feature
            np.save(save_path/run_id/f'{frame_num}.npy', ft)
    
    np.save(save_path/'center.npy', avg_latent_feature)
            