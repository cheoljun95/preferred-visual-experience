from .vae import VAE
import sys
sys.path.append("..")
try:
    from utils.visualize import *
except:
    pass
import torch
import torch.nn as nn
try:
    from pytorch_lightning import LightningModule
except:
    LightningModule = nn.Module


def anneal_coef(step, anneal_step_range, anneal_value_range):

    step = min(max(anneal_step_range[0], step), anneal_step_range[1])
    ratio = (step - anneal_step_range[0]) / (anneal_step_range[1] - anneal_step_range[0])
    annealed = anneal_value_range[0] - (anneal_value_range[0] - anneal_value_range[1]) * ratio

    return annealed


class DrivingSceneVAE(LightningModule):
    def __init__(
            self,
            vae_configs,
            loss_coef_instructions,
            lr = 0.001,
            use_cosine_lr=False,
            T_max=None,
            use_halfprecision=False,
            **kwargs,
    ):

        super().__init__()
        self.lr = lr
        self.net = VAE(**vae_configs)
        self.use_halfprecision=use_halfprecision
        if self.use_halfprecision:
            self.net = self.net.half()
        self.loss_coef_dict = loss_coef_instructions
        self.non_val_loss = []
        self.inter_cnt_ = 0
        self.use_cosine_lr = use_cosine_lr
        self.T_max = T_max

    def _get_latent_feats(self, imgs):
        outputs = self.net._get_latent(imgs)
        return outputs

    def forward(self, x):
        return self.net(x)

    def _update_coef(self):
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if 'anneal_step_range' in coef_inst.keys() and coef_inst['anneal_step_range'] is not None:
                coef_inst['value'] = anneal_coef(self.global_step, coef_inst['anneal_step_range'],
                                                 coef_inst['anneal_value_range'])
                self.log(f'{coef_name}_coef', coef_inst['value'])

    def training_step(self, batch, batch_idx,optimizer_idx=None):
        imgs = batch['images']
        imgs = torch.transpose(imgs, 1, 2) # so imgs: (batch,channel,seq,...)
        if self.use_halfprecision:
            imgs =imgs.half()
        outputs = self(imgs)

        loss_val =  0
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys():
                loss_val += coef_inst['value']*outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name])
        self._update_coef()
        self.log('train_loss',loss_val)

        return loss_val

    def validation_step(self, batch, batch_idx):
        imgs = batch['images']
        imgs = torch.transpose(imgs, 1, 2) # so imgs: (batch,channel,seq,...)
        if self.use_halfprecision:
            imgs =imgs.half()
        outputs = self(imgs)

        loss_val =  0
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys():
                if coef_name not in self.non_val_loss:
                    loss_val += coef_inst['value'] * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name])
        self.log(f'val_loss', loss_val)


        batch_size= imgs.shape[0]
        seq_len = imgs.shape[2]

        log_dict = {'val_loss':loss_val}
        if (batch_idx % 100)==0:

            sample_idx = int( self.inter_cnt_ % batch_size)
            self.inter_cnt_ += 1
            retrieved_images = []
            for i in range(seq_len):
                orig_recon_img = [inverse_transform(imgs[sample_idx][:,i,:,:]),
                                  inverse_transform(outputs['recon'][sample_idx][:,i,:,:]),]
                retrieved_images.append(np.concatenate(orig_recon_img,0))

            retrieved_images = np.concatenate(retrieved_images,1)

            self.logger.experiment.add_image('check', retrieved_images,
                                              self.global_step*1000 + batch_idx, dataformats='HWC')

        return log_dict


    def configure_optimizers(self):

        opt_fun = torch.optim.Adam
        opt = opt_fun(self.net.parameters(),lr=self.lr,eps=1e-4 if self.use_halfprecision else 1e-8)

        if self.use_cosine_lr:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,eta_min=self.lr*.1,T_max=self.T_max)
            return [opt], [{"scheduler": sch, "interval": "step"}]
        else:
            return [opt], []
