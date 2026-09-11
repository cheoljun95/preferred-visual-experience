from dataset.datamodule import DrivingSeqDataModule
from model.drivingscene_vae import DrivingSceneVAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint,EarlyStopping, Callback
import hydra
import torch
from utils.utils import load_model

@hydra.main(config_path='configs', default='vae-2sec')
def main(cfg):
    
    if 'seed' in cfg.keys():
        torch.manual_seed(int(cfg['seed']))
    
    vae_configs = cfg['model_configs']['vae_configs']
        
    # datamodule
    datamodule = DrivingSeqDataModule(resize=[vae_configs['input_shape'][1],
                                              vae_configs['input_shape'][2]], **cfg['data_configs'])
    
    if 'load_action' in cfg['data_configs'] and cfg['data_configs']['load_action']:
        DrivingSceneVAE = DrivingSceneActionVAE
    else:
        from model.drivingscene_vae import DrivingSceneVAE
        
    # model
    model = DrivingSceneVAE(**cfg['model_configs'])
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpoint(
        every_n_epochs=cfg['checkpoint_epoch'],
        #save_weights_only=True
    )

    # Trainer
    if cfg['gpus'] is not None:
        gpus = [int(x) for x in cfg['gpus'].split(',')]
    else:
        gpus= None

    callbacks  = [#checkpoint_callback_topk, 
            checkpoint_callback_by_epoch,LearningRateMonitor(logging_interval='step')]

    if 'experiment_name' in cfg.keys() and cfg['experiment_name'] is not None:
        save_dir = cfg['experiment_name']
    else:
        save_dir = None
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         strategy="ddp",
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         val_check_interval=cfg['val_check_interval'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         accumulate_grad_batches=cfg['accumulate_grad_batches'],
                         gradient_clip_val=0.5,
                         default_root_dir=save_dir,
                        
                        )

    # fit model
    trainer.fit(model,datamodule, ckpt_path=cfg['resume_ckpt_path'])

if __name__ =='__main__':
    main()
