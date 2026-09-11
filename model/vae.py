import torch
from torch import nn
from torch.nn import functional as F
from .block import ResBlock, ResUpBlock
import numpy as np

def soft_clamp(x: torch.Tensor, temp=0.2):
    return x.div(1/temp).tanh_().mul(1/temp)

def kld(mu, logvar, pmu=0, plogvar=0):
    return 0.5 * (plogvar- logvar + torch.exp(logvar-plogvar) + ((mu-pmu)/torch.exp(0.5*plogvar)).pow(2) - 1.0).mean()

class Encoder(nn.Module):
    '''
    3D CNN Encoder
    '''
    def __init__(self,
                 input_channel,
                 channels,
                 strides,
                 temporal_strides,
                 kernel_sizes=None,
                 temporal_kernel_sizes=None,
                 use_avgpool = False,
                 do_avgpool = True,
                 use_residual = True,
                 use_spectralnorm= False,
                 use_film=[],
                 hyp_dim=None,
                 post_downsample=True,
                 ):
        super().__init__()

        self.do_avgpool = do_avgpool
        channels = [input_channel, *channels]
        strides = [1, *strides]
        temporal_strides = [1, *temporal_strides]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)
        if temporal_kernel_sizes is None:
            temporal_kernel_sizes = [1, *([3] * (len(channels) - 1))]

        layers = []

        for i in range(0, len(channels) - 1):
            layers.append(ResBlock(num_ins=channels[i],
                                   num_outs=channels[i + 1],
                                   stride=strides[i],
                                   kernel_size=kernel_sizes[i],
                                   temporal_stride=temporal_strides[i],
                                   temporal_kernel_size=temporal_kernel_sizes[i],
                                   use_spectralnorm=use_spectralnorm,
                                   use_film=(i in use_film),
                                   hyp_dim=hyp_dim,
                                   use_res=use_residual,
                                   post_downsample=post_downsample,
                                   use_avgpool=use_avgpool))


        self.layers = nn.ModuleList(layers)
        if self.do_avgpool:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = None

    def forward(self, x, hyp=None):

        for i,l in enumerate(self.layers):
            x = l(x,hyp)
        if self.do_avgpool:
            return self.avgpool(x)
        else:
            return x

    def _get_latent_dimension(self, x):
        dimensions = []
        for l in self.layers:
            x = l(x)
            dimensions.append(x.shape[1:])
        return dimensions


class Decoder(nn.Module):
    '''
    3D CNN Decoder
    '''
    def __init__(self,
                 output_channel,
                 start_channel,
                 channels,
                 upsample_sizes,
                 kernel_sizes=None,
                 temporal_kernel_sizes=None,
                 upsample_method='trilinear',
                 use_residual = True,
                 ):
        super().__init__()

        self.upsample_sizes = upsample_sizes
        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)
        if temporal_kernel_sizes is None:
            temporal_kernel_sizes = [3] * (len(channels))

        layers = []
        for i in range(1, len(upsample_sizes)):
            layers.append(ResUpBlock(upsample_sizes[i],
                                             channels[i - 1],
                                             channels[i],
                                             kernel_sizes[i],
                                             temporal_kernel_sizes[i],
                                     upsample_method=upsample_method,
                                     use_res=use_residual,
                                     input_size=upsample_sizes[i-1]
                                     ))


        self.linear = nn.Linear(start_channel, channels[0]*np.array(upsample_sizes[0]).prod())
        self.to_rgb = nn.Sequential(
                        nn.BatchNorm3d(channels[-1]),
                        nn.ReLU(),
                        nn.Conv3d(channels[-1], output_channel, 3, 1, 1),
                        nn.Tanh()
                    )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        b = x.shape[0]
        x = self.linear(x.view(b,-1)).view(b,-1,*self.upsample_sizes[0])
        for i,l in enumerate(self.layers):

            x = l(x)
        x = self.to_rgb(x)
        return x

class VAE(nn.Module):

    def __init__(self,
                 data_channel,
                 input_shape,
                 channels,
                 strides,
                 temporal_strides,
                 output_shape= None,
                 skip_layer = 1,
                 full_temporal_dim=False,
                 no_temporal_kernel=False,
                 use_avgpool = False,
                 use_residual = True,
                 use_spectralnorm=False,
                 do_avgpool=False,
                 post_downsample=False,
                 use_film=[],
                 hyp_dim=None,
                 upsample_method='trilinear',
                 use_residual_dec= True,
                 use_batchnorm_on_latent=False,
                 use_spectralnorm_on_latent=False,
                 trainable_prior=False,
                 **kwargs,
                ):
        super().__init__()

        self.do_avgpool = do_avgpool
        if no_temporal_kernel:
            temporal_kernel_sizes = [1]*(len(channels)+1)
        else:
            temporal_kernel_sizes = None
        self.encoder = Encoder(input_channel = data_channel,
                               channels=channels,
                               strides=strides,
                               temporal_strides=temporal_strides,
                               use_avgpool=use_avgpool,
                               use_residual=use_residual,
                               temporal_kernel_sizes=temporal_kernel_sizes,
                               use_spectralnorm=use_spectralnorm,
                               use_film=use_film,
                               hyp_dim=hyp_dim,
                               post_downsample=post_downsample,
                               do_avgpool=do_avgpool
                               )

        with torch.no_grad():
            x = torch.rand(1, data_channel, *input_shape)
            dims = self.encoder._get_latent_dimension(x)

        self.latent_dim = np.array(dims[-1])
        print(self.latent_dim)
        if output_shape is None:
            upsample_sizes = [x.shape[2:]]
            temp_factor = 1
        else:
            upsample_sizes = [[int(output_shape[0]),
                              int(output_shape[1]),
                              int(output_shape[2])]]
            temp_factor = int(output_shape[0]/input_shape[0]) # assume the factor is natural number.
        decoder_channels = [dims[0][0], dims[0][0]]
        for i in range(1, len(dims)-skip_layer):
            decoder_channels.append(dims[i][0])
            if full_temporal_dim:
                upsample_sizes.append([input_shape[0],dims[i][2],dims[i][3]])
            else:
                upsample_sizes.append([dims[i][1]*temp_factor,dims[i][2],dims[i][3]])
        upsample_sizes.reverse()
        decoder_channels.reverse()
        start_channel = dims[-1][0]
        self.decoder = Decoder(data_channel,
                               start_channel,
                               decoder_channels,
                               upsample_sizes,
                               temporal_kernel_sizes=temporal_kernel_sizes,
                               upsample_method=upsample_method,
                               use_residual=use_residual_dec,
                              )

        if use_batchnorm_on_latent:
            self.fc_mu = nn.Sequential(nn.Conv3d(self.latent_dim[0],self.latent_dim[0],kernel_size=1,
                                         stride=1,padding=0),
                                       nn.BatchNorm3d(self.latent_dim[0]))
        else:
            self.fc_mu = nn.Conv3d(self.latent_dim[0],self.latent_dim[0],kernel_size=1,
                                     stride=1,padding=0)
        if use_spectralnorm_on_latent:
            assert not use_batchnorm_on_latent
            self.fc_mu = nn.utils.spectral_norm(self.fc_mu)
        self.fc_std = nn.Conv3d(self.latent_dim[0], self.latent_dim[0], kernel_size=1,
                                     stride=1, padding=0)

        self.mu_prior = nn.parameter.Parameter(torch.zeros(self.latent_dim[0], 1, 1, 1), requires_grad=trainable_prior)
        self.mu_logvar = nn.parameter.Parameter(torch.zeros(self.latent_dim[0], 1, 1, 1), requires_grad=trainable_prior)



    def _get_latent(self,x):
        latent_feats = self.encoder(x) #(B,d,1,1,1)
        mu = self.fc_mu(latent_feats)
        return mu.squeeze(-1).squeeze(-1).squeeze(-1)

    def forward(self, x):
        latent_feats = self.encoder(x) #(B,d,1,1,1)
        mu = self.fc_mu(latent_feats)
        log_var = self.fc_std(latent_feats)
        p, q, z = self.sample(mu, log_var)

        outputs = {}
        outputs['recon'] = self.decoder(z)
        outputs['recon_loss'] = F.mse_loss(outputs['recon'], x, reduction = "mean")
        outputs['kld_loss'] = kld(mu, log_var, self.mu_prior, self.mu_logvar)

        return outputs

    def sample(self, mu, log_var):
        std = torch.exp(soft_clamp(log_var))
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(soft_clamp(mu), std)
        z = q.rsample()
        return p, q, z



