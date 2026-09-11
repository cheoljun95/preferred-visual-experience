import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ResBlock(nn.Module):
    '''
    3D and temporal convolution is causal.
    '''

    def __init__(self, num_ins, num_outs, stride=1, kernel_size=3, temporal_stride=1, temporal_kernel_size=3, use_spectralnorm=False,
                 use_film=False, hyp_dim=16,use_res=True,
                use_avgpool=False, post_downsample=True):
        super().__init__()

        self.temporal_stride = temporal_stride
        self.temporal_kernel_size = temporal_kernel_size
        if use_avgpool:
            self.pool = nn.AvgPool3d([temporal_stride,stride,stride],[temporal_stride,stride,stride])
            self.conv1 = nn.Conv3d(num_ins, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                          padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                          stride=[1, 1, 1])

            self.conv2 = nn.Conv3d(num_outs, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                      padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                      stride=[1, 1, 1])
        else:
            self.pool = None
            if post_downsample:
                self.conv1 = nn.Conv3d(num_ins, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                          padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                          stride=[temporal_stride, stride, stride])

                self.conv2 = nn.Conv3d(num_outs, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                          padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                          stride=[1, 1, 1])

            else:
                self.conv1 = nn.Conv3d(num_ins, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                          padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                          stride=[temporal_stride, stride, stride])

                self.conv2 = nn.Conv3d(num_outs, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                          padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2],
                                          stride=[1, 1, 1])


        if use_spectralnorm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.bn1 = None
            self.bn2 = None
        else:
            self.bn1 = nn.BatchNorm3d(num_outs)
            self.bn2 = nn.BatchNorm3d(num_outs)

        if use_res and (stride != 1 or num_ins != num_outs or temporal_stride != 1):
            if use_avgpool:
                self.residual_path = nn.Conv3d(num_ins, num_outs, kernel_size=[1, 1, 1])
            else:
                self.residual_path = nn.Conv3d(num_ins, num_outs, kernel_size=[1, 1, 1],
                                           stride=[temporal_stride, stride, stride])
            if use_spectralnorm:
                self.residual_path = nn.utils.spectral_norm(self.residual_path)
                self.res_norm = None
            else:
                self.res_norm = nn.BatchNorm3d(num_outs)
        else:
            self.residual_path = None
        self.use_res = use_res
        self.act = nn.ELU(inplace=True)
        self.use_film = use_film
        if self.use_film:
            self.mu = nn.Linear(hyp_dim,num_outs)
            self.sigma = nn.Linear(hyp_dim,num_outs)

    def forward(self, x, hyp=None):
        input_value = x
        x = self.conv1(x)

        if self.bn1 is not None and x.shape[-1] > 1 and x.shape[-2] > 1 and x.shape[-3] > 1:
            x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.bn2 is not None and x.shape[-1] > 1 and x.shape[-2] > 1 and x.shape[-3] > 1:
            x = self.bn2(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.residual_path is not None:
            res = self.residual_path(input_value)
            if self.res_norm is not None and res.shape[-1] > 1 and res.shape[-2] > 1 and res.shape[-3] > 1:
                res = self.res_norm(res)
            if self.pool is not None:
                res = self.pool(res)
        else:
            res = input_value
        if self.use_film:
            if hyp is None:
                print('hyp is None!')
            else:
                x = self.mu(hyp).view(x.shape[0],x.shape[1],1,1,1) + self.sigma(hyp).view(x.shape[0],x.shape[1],1,1,1)*x
        if self.use_res:
            x = self.act(x + res)
        else:
            x= self.act(x)
        return x


class ResUpBlock(nn.Module):
    '''
    3D upsample & convolution with residual connection
    '''
    def __init__(self, upsample_size, num_ins, num_outs, kernel_size=3, temporal_kernel_size=3,
                 upsample_method='trilinear', use_res=True, input_size=None):
        super().__init__()

        assert upsample_method in ['trilinear', 'transpose']
        if upsample_method == 'trilinear' :
            self.upsample = nn.Upsample(size=upsample_size, mode='trilinear', align_corners=True)
        elif upsample_method=='transpose':
            def get_config(insize,outsize):
                if outsize > insize:
                    kernel = 2
                    stride = 2
                    if (2*insize-outsize)%2 == 1:
                        output_padding = 1
                    else:
                        output_padding = 0
                    input_padding = int((2*insize-outsize+output_padding)/2)
                else:
                    stride = 1
                    kernel = 1
                    input_padding = 0
                    output_padding = 0
                return stride, kernel, input_padding, output_padding

            s_t,k_t,ip_t,op_t = get_config(input_size[0],upsample_size[0])
            s_h,k_h,ip_h,op_h = get_config(input_size[1],upsample_size[1])
            s_w,k_w,ip_w,op_w = get_config(input_size[2],upsample_size[2])

            self.upsample = nn.ConvTranspose3d(in_channels=num_ins,
                                               out_channels=num_ins,
                                               kernel_size=[k_t,k_h,k_w],
                                               stride=[s_t,s_h,s_w],
                                               padding=[ip_t,ip_h,ip_w],
                                               output_padding=[op_t,op_h,op_w])
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv3d(num_ins, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                  padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2])
        self.conv2 = nn.Conv3d(num_outs, num_outs, kernel_size=[temporal_kernel_size, kernel_size, kernel_size],
                                  padding=[temporal_kernel_size//2, kernel_size // 2, kernel_size // 2])
        if use_res:
            self.residual_path = nn.Sequential(
                nn.Upsample(size=upsample_size, mode='trilinear', align_corners=True),
                nn.Conv3d(num_ins, num_outs, kernel_size=1, padding=0),
                nn.BatchNorm3d(num_outs)
            )
        else:
            self.residual_path = None

        self.bn1 = nn.BatchNorm3d(num_outs)
        self.bn2 = nn.BatchNorm3d(num_outs)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        input_value = x
        x = self.upsample(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)

        if self.residual_path is not None:
            x = x+self.residual_path(input_value)
        return self.act(x)
