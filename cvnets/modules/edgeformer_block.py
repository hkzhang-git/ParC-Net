from torch import nn, Tensor
import math
import torch
import numpy as np
from torch.nn import functional as F
from typing import Optional, Dict, Tuple

from .transformer import TransformerEncoder
from .base_module import BaseModule
from ..misc.profiler import module_profile
from ..layers import ConvLayer, get_normalization_layer, get_activation_fn, Dropout
from ..modules import InvertedResidual


class shuffle_layer(nn.Module):
    def __init__(self, input_channel, groups=4):
        super(shuffle_layer, self).__init__()
        ind = np.array(list(range(input_channel)))
        ind = ind.reshape(groups, input_channel//groups)
        ind = ind.T
        self.shuffle_ind = ind.reshape(input_channel)

    def forward(self, x):
        out = x[:, self.shuffle_ind]
        return out

# channel wise attention
class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.Hardswish(),
            nn.Conv2d(channel//reduction, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x*y.expand_as(x)

class channel_pooling(nn.Module):
    def forward(self,x):
        max_res = torch.mean(x, dim=1, keepdim=True)
        avg_res = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((max_res, avg_res), dim=1)

# spatial attention
class SA_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA_layer, self).__init__()
        self.gcp = channel_pooling()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(kernel_size, kernel_size), padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Hardswish(),
            nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), padding=(kernel_size-1)//2, bias=False),
            nn.Hardsigmoid(),
        )
    def forward(self, x):
        p_x = self.gcp(x)
        return x*self.conv(p_x)

# global circular conv, where dynamic kernel is used
class gcc_dk(nn.Module):
    def __init__(self, channel, direction='H'):
        super(gcc_dk, self).__init__()
        self.direction = direction
        self.gap = nn.AdaptiveAvgPool2d(1)

        if direction=='H':
            self.kernel_generate_conv = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=channel),
                nn.BatchNorm2d(channel),
                nn.Hardswish(),
                nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=channel),
            )
        elif direction=='W':
            self.kernel_generate_conv = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=channel),
                nn.BatchNorm2d(channel),
                nn.Hardswish(),
                nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=channel),
            )

    def forward(self, x):
        glob_info = self.gap(x)
        if self.direction=='H':
            H_info = torch.mean(x, dim=3, keepdim=True)
            kernel_input = H_info + glob_info.expand_as(H_info)
        elif self.direction=='W':
            W_info = torch.mean(x, dim=2, keepdim=True)
            kernel_input = W_info + glob_info.expand_as(W_info)

        kernel_weight = self.kernel_generate_conv(kernel_input)

        return kernel_weight

##*******************************************************************************************************big kernel conv
# big kernel conv based pure ConvNet Meta-Former block
class bkc_mf_block(BaseModule):
    def __init__(self, dim: int, big_kernel_size: int, mid_mix: Optional[bool]=True, bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2, ffn_dropout=0.0, dropout=0.1):

        super(bkc_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.bk_conv_1_H = nn.Conv2d(dim, dim, kernel_size=(big_kernel_size, 1), padding=(big_kernel_size//2, 0), groups=dim, bias=bias)
        self.bk_conv_1_W = nn.Conv2d(dim, dim, kernel_size=(1, big_kernel_size), padding=(0, big_kernel_size//2), groups=dim, bias=bias)

        self.bk_conv_2_W = nn.Conv2d(dim, dim, kernel_size=(big_kernel_size, 1), padding=(big_kernel_size//2, 0), groups=dim, bias=bias)
        self.bk_conv_2_H = nn.Conv2d(dim, dim, kernel_size=(1, big_kernel_size), padding=(0, big_kernel_size//2), groups=dim, bias=bias)

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix

        # channel part

        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2

        #************************************************************************************************** spatial part
        # pre norm
        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = self.bk_conv_1_H(x_1)
        x_2_1 = self.bk_conv_1_W(x_2)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        # stage 2
        x_1_2 = self.bk_conv_2_W(x_1_1)
        x_2_2 = self.bk_conv_2_H(x_2_1)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        #************************************************************************************************** channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ffn(x_ffn)

        return x_ffn

# big kernel conv based pure ConvNet Meta-Former block
# channel wise attention is used
class bkc_ca_mf_block(BaseModule):
    def __init__(self, dim: int, big_kernel_size: int, mid_mix: Optional[bool]=True, bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2, ffn_dropout=0.0, dropout=0.1):

        super(bkc_ca_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.bk_conv_1_H = nn.Conv2d(dim, dim, kernel_size=(big_kernel_size, 1), padding=(big_kernel_size//2, 0), groups=dim, bias=bias)
        self.bk_conv_1_W = nn.Conv2d(dim, dim, kernel_size=(1, big_kernel_size), padding=(0, big_kernel_size//2), groups=dim, bias=bias)

        self.bk_conv_2_W = nn.Conv2d(dim, dim, kernel_size=(big_kernel_size, 1), padding=(big_kernel_size//2, 0), groups=dim, bias=bias)
        self.bk_conv_2_H = nn.Conv2d(dim, dim, kernel_size=(1, big_kernel_size), padding=(0, big_kernel_size//2), groups=dim, bias=bias)

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix

        # channel part

        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

        self.ca = CA_layer(channel=2 * dim)

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2

        #************************************************************************************************** spatial part
        # pre norm
        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = self.bk_conv_1_H(x_1)
        x_2_1 = self.bk_conv_1_W(x_2)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        # stage 2
        x_1_2 = self.bk_conv_2_W(x_1_1)
        x_2_2 = self.bk_conv_2_H(x_2_1)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        #************************************************************************************************** channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ca(self.ffn(x_ffn))

        return x_ffn

##*************************************************************************************************global circular conv
# global circular conv based pure ConvNet Meta-Former block
class gcc_mf_block(BaseModule):
    def __init__(self,
                 dim: int,
                 meta_kernel_size: int,
                 instance_kernel_method='crop',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super(gcc_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.meta_kernel_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.meta_kernel_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.meta_1_H_bias = None
            self.meta_1_W_bias = None
            self.meta_2_H_bias = None
            self.meta_2_W_bias = None

        self.instance_kernel_method = instance_kernel_method

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))


        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim

        # channel part

        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

    def get_instance_kernel(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_kernel_1_H[:, :, : instance_kernel_size,:], \
                   self.meta_kernel_1_W[:, :, :, :instance_kernel_size], \
                   self.meta_kernel_2_H[:, :, :instance_kernel_size, :], \
                   self.meta_kernel_2_W[:, :, :, :instance_kernel_size]

        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape = [instance_kernel_size, 1]
            W_shape = [1, instance_kernel_size]
            return F.interpolate(self.meta_kernel_1_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_1_W, W_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_W, W_shape, mode='bilinear', align_corners=True),

        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2
        _, _, f_s, _ = x_1.shape

        K_1_H, K_1_W, K_2_H, K_2_W = self.get_instance_kernel(f_s)

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = F.conv2d(torch.cat((x_1, x_1[:, :, :-1, :]), dim=2), weight=K_1_H, bias=self.meta_1_H_bias, padding=0,
                         groups=self.dim)
        x_2_1 = F.conv2d(torch.cat((x_2, x_2[:, :, :, :-1]), dim=3), weight=K_1_W, bias=self.meta_1_W_bias, padding=0,
                         groups=self.dim)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3), weight=K_2_W, bias=self.meta_2_W_bias,
                         padding=0, groups=self.dim)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), weight=K_2_H, bias=self.meta_2_H_bias,
                         padding=0, groups=self.dim)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ffn(x_ffn)

        return x_ffn

# global circular conv based pure ConvNet Meta-Former block
# channel wise attention is used
class gcc_ca_mf_block(BaseModule):
    def __init__(self,
                 dim: int,
                 meta_kernel_size: int,
                 instance_kernel_method='crop',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super(gcc_ca_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.meta_kernel_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.meta_kernel_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.meta_1_H_bias = None
            self.meta_1_W_bias = None
            self.meta_2_H_bias = None
            self.meta_2_W_bias = None

        self.instance_kernel_method = instance_kernel_method

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))


        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim

        # channel part
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

        self.ca = CA_layer(channel=2*dim)

    def get_instance_kernel(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_kernel_1_H[:, :, : instance_kernel_size,:], \
                   self.meta_kernel_1_W[:, :, :, :instance_kernel_size], \
                   self.meta_kernel_2_H[:, :, :instance_kernel_size, :], \
                   self.meta_kernel_2_W[:, :, :, :instance_kernel_size]

        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape = [instance_kernel_size, 1]
            W_shape = [1, instance_kernel_size]
            return F.interpolate(self.meta_kernel_1_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_1_W, W_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_W, W_shape, mode='bilinear', align_corners=True),

        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2
        _, _, f_s, _ = x_1.shape

        K_1_H, K_1_W, K_2_H, K_2_W = self.get_instance_kernel(f_s)

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = F.conv2d(torch.cat((x_1, x_1[:, :, :-1, :]), dim=2), weight=K_1_H, bias=self.meta_1_H_bias, padding=0,
                         groups=self.dim)
        x_2_1 = F.conv2d(torch.cat((x_2, x_2[:, :, :, :-1]), dim=3), weight=K_1_W, bias=self.meta_1_W_bias, padding=0,
                         groups=self.dim)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3), weight=K_2_W, bias=self.meta_2_W_bias,
                         padding=0, groups=self.dim)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), weight=K_2_H, bias=self.meta_2_H_bias,
                         padding=0, groups=self.dim)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ca(self.ffn(x_ffn))

        return x_ffn

# global circular conv based pure ConvNet Meta-Former block
# dynamic kernel is used to handle the problem that inputs have different spatial resolution
class gcc_dk_mf_block(BaseModule):
    def __init__(self,
                 dim: int,
                 meta_kernel_size: int,
                 instance_kernel_method='crop',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super(gcc_dk_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.instance_kernel_method = instance_kernel_method

        self.kernel_generate_1_H = gcc_dk(dim, 'H')
        self.kernel_generate_1_W = gcc_dk(dim, 'W')
        self.kernel_generate_2_H = gcc_dk(dim, 'H')
        self.kernel_generate_2_W = gcc_dk(dim, 'W')

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.meta_1_H_bias = None
            self.meta_1_W_bias = None
            self.meta_2_H_bias = None
            self.meta_2_W_bias = None

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim

        # channel part
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )


    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2
        b, c, f_s, _ = x_1.shape

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1

        x_1_r, x_2_r = x_1.view(1, b*c, f_s, f_s), x_2.view(1, b*c, f_s, f_s)
        x_1_1 = F.conv2d(torch.cat((x_1_r, x_1_r[:, :, :-1, :]), dim=2), weight=self.kernel_generate_1_H(x_1).view(b*c, 1, f_s, 1),
                         bias=self.meta_1_H_bias, padding=0, groups=b * c)
        x_1_1 = F.instance_norm(x_1_1)

        x_2_1 = F.conv2d(torch.cat((x_2_r, x_2_r[:, :, :, :-1]), dim=3), weight=self.kernel_generate_1_W(x_2).view(b*c, 1, 1, f_s),
                         bias=self.meta_2_W_bias, padding=0, groups=b * c)
        x_2_1 = F.instance_norm(x_2_1)

        x_1_1, x_2_1 = x_1_1.view(b, c, f_s, f_s), x_2_1.view(b, c, f_s, f_s)

        # mid_mix is not used in here.
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_1_r, x_2_1_r = x_1_1.view(1, b*c, f_s, f_s), x_2_1.view(1, b*c, f_s, f_s)

        x_1_2 = F.conv2d(torch.cat((x_1_1_r, x_1_1_r[:, :, :, :-1]), dim=3), weight=self.kernel_generate_2_W(x_1_1).view(b*c, 1, 1, f_s),
                         bias=self.meta_2_W_bias, padding=0, groups=b * c)
        x_1_2 = F.instance_norm(x_1_2)

        x_2_2 = F.conv2d(torch.cat((x_2_1_r, x_2_1_r[:, :, :-1, :]), dim=2), weight=self.kernel_generate_2_H(x_2_1).view(b*c, 1, f_s, 1),
                         bias=self.meta_2_H_bias, padding=0, groups=b * c)
        x_2_2 = F.instance_norm(x_2_2)

        x_1_2, x_2_2 = x_1_2.view(b, c, f_s, f_s), x_2_2.view(b, c, f_s, f_s)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ffn(x_ffn)

        return x_ffn

# global circular conv based pure ConvNet Meta-Former block
# dynamic kernel is used to handle the problem that inputs have different spatial resolution
class gcc_dk_ca_mf_block(BaseModule):
    def __init__(self,
                 dim: int,
                 meta_kernel_size: int,
                 instance_kernel_method='crop',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super(gcc_dk_ca_mf_block, self).__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.kernel_generate_1_H = gcc_dk(dim, 'H')
        self.kernel_generate_1_W = gcc_dk(dim, 'W')
        self.kernel_generate_2_H = gcc_dk(dim, 'H')
        self.kernel_generate_2_w = gcc_dk(dim, 'W')

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.meta_1_H_bias = None
            self.meta_1_W_bias = None
            self.meta_2_H_bias = None
            self.meta_2_W_bias = None

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim

        # channel part
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

        self.ca = CA_layer(channel=2*dim)


    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2
        b, c, f_s, _ = x_1.shape

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1

        x_1_r, x_2_r = x_1.view(1, b*c, f_s, f_s), x_2.view(1, b*c, f_s, f_s)
        x_1_1 = F.conv2d(torch.cat((x_1_r, x_1_r[:, :, :-1, :]), dim=2), weight=self.kernel_generate_1_H(x_1_r),
                         bias=self.meta_1_H_bias, padding=0, groups=b * c)
        x_2_1 = F.conv2d(torch.cat((x_2_r, x_2_r[:, :, :, :-1]), dim=3), weight=self.kernel_generate_1_W(x_2_r),
                         bias=self.meta_2_W_bias, padding=0, groups=b * c)

        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3), weight=self.kernel_generate_2_W(x_1_1),
                         bias=self.meta_2_W_bias, padding=0, groups=b * c)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), weight=self.kernel_generate_2_H(x_2_1),
                         bias=self.meta_2_H_bias, padding=0, groups=b * c)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ca(self.ffn(x_ffn))

        return x_ffn


block_dict={
    'bkc': bkc_mf_block,
    'bkc_ca': bkc_ca_mf_block,
    'gcc': gcc_mf_block,
    'gcc_ca': gcc_ca_mf_block,
    'gcc_dk': gcc_dk_mf_block,
    'gcc_dk_ca': gcc_dk_ca_mf_block
}


# outer_frame_v1, the outer frame adopted in paper "EdgeFormer:Improving Light-weight ConvNets by Learning from Vision Transformers"
# mobilevit like structure. For each stage, several vit blocks and one mobilenetv2(mb2) block is used.
#                        ---C---   ---C---   ---C---
# outer structure C->C-> C-T-T-T-> C-T-T-T-> C-T-T-T->

class outer_frame_v1(BaseModule):
    def __init__(self, opts,
                 meta_kernel_size: int,
                 in_channels: int,
                 cf_s_channels: int,
                 n_blocks:int,
                 big_kernel_size: int,
                 meta_encoder: Optional[str]='gcc_ca',
                 instance_kernel_method: Optional[str]='crop',
                 fusion_method: Optional[str]='add',
                 use_pe: Optional[bool] = True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=False,
                 cf_ffn_channels: Optional[int]=2,
                 ffn_dropout: Optional[float]=0.0,
                 dropout:Optional[float]=0.1,
                 dilation: Optional[int] = 1):

        super(outer_frame_v1, self).__init__()

        # structure parameters for computing madds
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.cf_s_channels = cf_s_channels
        self.cf_ffn_channels = cf_ffn_channels
        self.big_kernel_size = big_kernel_size
        self.meta_encoder = meta_encoder
        self.fusion_method = fusion_method
        self.bias = bias
        self.use_pe = use_pe

        # **************************************************************************************dim adaptive, local rep
        self.dim_adaptive=nn.Sequential(
            ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=cf_s_channels,
                kernel_size=3, stride=1, use_norm=True, use_act=True, dilation=dilation, groups=np.gcd(cf_s_channels, in_channels)
            ),
            ConvLayer(
                opts=opts, in_channels=cf_s_channels, out_channels=cf_s_channels,
                kernel_size=1, stride=1, use_norm=False, use_act=False
            )
        )
        # **************************************************************************************edge-former, global rep
        spatial_global_conv = []

        if meta_encoder.startswith('bkc'):
            bk_block = block_dict[meta_encoder]
            for block_idx in range(n_blocks):
                spatial_global_conv.append(
                    bk_block(dim=cf_s_channels // 2,
                             big_kernel_size=big_kernel_size,
                             mid_mix=mid_mix,
                             bias=bias,
                             ffn_dim=cf_ffn_channels,
                             ffn_dropout=ffn_dropout,
                             dropout=dropout
                             )
                )

        elif meta_encoder.startswith('gcc'):
            gcc_block = block_dict[meta_encoder]
            for block_idx in range(n_blocks):
                # note that, position embedding is just used in the first gcc block of each stage.
                # We can inserted it into each gcc block, but we have not tried this yet.
                add_pe = use_pe if block_idx==0 else False
                spatial_global_conv.append(
                    gcc_block(dim=cf_s_channels // 2,
                              meta_kernel_size=meta_kernel_size,
                              instance_kernel_method=instance_kernel_method,
                              use_pe=add_pe,
                              mid_mix=mid_mix,
                              bias=bias,
                              ffn_dim=cf_ffn_channels,
                              ffn_dropout=ffn_dropout,
                              dropout=dropout
                              )
                )
        else:
            print('{} is not supported !')

        spatial_global_conv.append(nn.BatchNorm2d(num_features=cf_s_channels))
        self.spatial_global_rep = nn.Sequential(*spatial_global_conv)


        # **************************************************************************************************fusion layer
        if fusion_method == 'add':
            self.channel_adaptive = ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels,
                                              groups=np.gcd(cf_s_channels, in_channels),
                                              kernel_size=1, stride=1, use_norm=True, use_act=True)
            self.local_global_fusion = ConvLayer(opts=opts, in_channels=in_channels, out_channels=in_channels,
                                                 kernel_size=3, stride=1, use_norm=True, use_act=True)

        elif fusion_method == 'concat':
            self.channel_adaptive = ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels,
                                              groups=np.gcd(cf_s_channels, in_channels),
                                              kernel_size=1, stride=1, use_norm=True, use_act=True)
            self.local_global_fusion = ConvLayer(opts=opts, in_channels=2*in_channels, out_channels=in_channels,
                                                 kernel_size=3, stride=1, use_norm=True, use_act=True)

        self.fusion_method = fusion_method

    def forward(self, x: Tensor) -> Tensor:
        res = x

        x = self.dim_adaptive(x)
        x = self.spatial_global_rep(x)

        x = self.channel_adaptive(x)

        if self.fusion_method == 'add':
            x = self.local_global_fusion(res + x)
        elif self.fusion_method == 'concat':
            x = self.local_global_fusion(torch.cat((x, res), dim=1))

        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = macs = 0.0
        b, c, h, w = input.shape

        # local rep
        out_da, p_da, m_da = module_profile(module=self.dim_adaptive, x=input)
        params, macs = params+p_da, macs+m_da

        # global rep
        p_global_rep = sum([p.numel() for p in self.spatial_global_rep.parameters()])

        if self.meta_encoder.startswith('bkc'):
            m_global_s = (self.big_kernel_size * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
            if self.bias:
                m_global_s += self.cf_s_channels * (h * w) * 2
            m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
                    self.cf_s_channels + self.cf_ffn_channels) * (h * w)

        elif self.meta_encoder.startswith('gcc'):
            m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
            if self.bias:
                m_global_s += self.cf_s_channels * (h * w) * 2
            m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
                    self.cf_s_channels + self.cf_ffn_channels) * (h * w)
        if self.meta_encoder.endswith('ca'):
            m_ca = self.cf_ffn_channels * (self.cf_ffn_channels // 16) * (h * w) * 2
        else:
            m_ca = 0

        m_global_rep = (m_global_s + m_global_ffn + m_ca) * self.n_blocks

        params, macs = params + p_global_rep, macs + m_global_rep

        # fusion part
        p_c_adaptive = sum([p.numel() for p in self.channel_adaptive.parameters()])
        m_c_adaptive = self.cf_s_channels * self.in_channels * (w * h) / np.gcd(self.cf_s_channels, self.in_channels)

        p_l_g_fusion = sum([p.numel() for p in self.local_global_fusion.parameters()])
        if self.fusion_method == 'add':
            m_l_g_fusion = 3*3*self.in_channels * self.in_channels * (w * h)
        elif self.fusion_method == 'concat':
            m_l_g_fusion = 3*3*2*self.in_channels * self.in_channels * (w * h)

        params, macs = params+p_c_adaptive+p_l_g_fusion, macs + m_c_adaptive + m_l_g_fusion

        return input, params, macs


# outer_frame_v2,
# coatnet like structure.
#                        -C-C-C-   -C-C-C-   -C-C-C-
# outer structure C->C-> -T-T-T->  -T-T-T--> -T-T-T-->
# this part may has some bugs. We have not tested this part
class outer_frame_v2(BaseModule):
    def __init__(self, opts,
                 meta_kernel_size: int,
                 in_channels: int,
                 cf_s_channels: int,
                 n_blocks:int,
                 big_kernel_size: int,
                 meta_encoder: Optional[str]='gcc_ca',
                 instance_kernel_method: Optional[str]='crop',
                 fusion_method: Optional[str]='add',
                 use_pe: Optional[bool] = True,
                 mid_mix: Optional[bool]=True,
                 bias: Optional[bool]=True,
                 cf_ffn_channels: Optional[int]=2,
                 ffn_dropout: Optional[float]=0.0,
                 dropout:Optional[float]=0.1,
                 dilation: Optional[int] = 1):

        super(outer_frame_v2, self).__init__()

        # structure parameters for computing madds
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.cf_s_channels = cf_s_channels
        self.cf_ffn_channels = cf_ffn_channels
        self.big_kernel_size = big_kernel_size
        self.meta_encoder = meta_encoder
        self.fusion_method = fusion_method
        self.bias = bias
        self.use_pe = use_pe

        #
        self.local_rep_channel = nn.ModuleList()
        self.global_rep_channel = nn.ModuleList()

        for i in range(n_blocks):
            self.local_rep_channel.append(InvertedResidual(opts=opts, in_channels=in_channels, out_channels=in_channels,
                                         stride=1, expand_ratio=2))

            if self.meta_encoder == 'bkc':
                self.global_rep_channel.append(
                    nn.Sequential(
                        # dim_adaptive
                        ConvLayer(
                            opts=opts, in_channels=in_channels, out_channels=cf_s_channels,
                            kernel_size=3, stride=1, use_norm=True, use_act=True, dilation=dilation,
                            groups=min(cf_s_channels, in_channels)
                        ),
                        ConvLayer(
                            opts=opts, in_channels=cf_s_channels, out_channels=cf_s_channels,
                            kernel_size=1, stride=1, use_norm=False, use_act=False
                        ),
                        # meta_former
                        bkc_mf_block(dim=cf_s_channels // 2,
                                           big_kernel_size=big_kernel_size,
                                           mid_mix=mid_mix,
                                           bias=bias,
                                           ffn_dim=cf_ffn_channels,
                                           ffn_dropout=ffn_dropout,
                                           dropout=dropout),
                        # channel adaptive
                        ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels, groups=cf_s_channels,
                                  kernel_size=1, stride=1, use_norm=True, use_act=True)
                    )
                )
            elif self.meta_encoder == 'gcc':
                self.global_rep_channel.append(
                    nn.Sequential(
                        ConvLayer(
                            opts=opts, in_channels=in_channels, out_channels=cf_s_channels,
                            kernel_size=3, stride=1, use_norm=True, use_act=True, dilation=dilation,
                            groups=cf_s_channels
                        ),
                        ConvLayer(
                            opts=opts, in_channels=cf_s_channels, out_channels=cf_s_channels,
                            kernel_size=1, stride=1, use_norm=False, use_act=False
                        ),
                        gcc_mf_block(dim=cf_s_channels // 2,
                                           meta_kernel_size=meta_kernel_size,
                                           instance_kernel_method=instance_kernel_method,
                                           use_pe=use_pe,
                                           mid_mix=mid_mix,
                                           bias=bias,
                                           ffn_dim=cf_ffn_channels,
                                           ffn_dropout=ffn_dropout,
                                           dropout=dropout),
                        ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels, groups=cf_s_channels,
                                  kernel_size=1, stride=1, use_norm=True, use_act=True)
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.n_blocks):
            x = self.local_rep_channel[i](x) + self.global_rep_channel[i](x)
        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = macs = 0.0
        b, c, h, w = input.shape

        for i in range(self.n_blocks):

            # local channel
            _, p_local, m_local = self.local_rep_channel[i].profile_module(input)
            # global channel

            p_global = sum([p.numel() for p in self.global_rep_channel[i].parameters()])
            m_dim_adaptive = (3*3) * (self.in_channels*self.cf_s_channels) * (h*w) / self.cf_s_channels + \
                             (1*1) * (self.cf_s_channels*self.cf_s_channels) * (h*w)

            if self.meta_encoder == 'big_kernel':
                m_global_s = (self.big_kernel_size * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
                if self.bias:
                    m_global_s += self.cf_s_channels * (h * w) * 2
                m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
                        self.cf_s_channels + self.cf_ffn_channels) * (h * w)

                m_global_rep = m_global_s + m_global_ffn

            elif self.meta_encoder == 'global_conv':
                m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
                if self.bias:
                    m_global_s += self.cf_s_channels * (h * w) * 2
                m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
                        self.cf_s_channels + self.cf_ffn_channels) * (h * w)

                m_global_rep = m_global_s + m_global_ffn
                if self.use_pe:
                    m_global_rep += c * h * w

            m_channel_adaptive = (1*1) * (self.cf_s_channels*self.in_channels) * (h*w) / self.cf_s_channels

            params = params + p_local + p_global
            macs = macs + m_local + m_dim_adaptive + m_global_rep + m_channel_adaptive + c*h*w

        return input, params, macs


# class cf_meta_v3(BaseModule):
#     def __init__(self, opts,
#                  meta_kernel_size: int,
#                  in_channels: int,
#                  cf_s_channels: int,
#                  n_blocks:int,
#                  big_kernel_size: int,
#                  meta_encoder: Optional[str]='big_kernel',
#                  instance_kernel_method: Optional[str]='crop',
#                  fusion_method: Optional[str]='add',
#                  use_pe: Optional[bool] = True,
#                  mid_mix: Optional[bool]=True,
#                  bias: Optional[bool]=True,
#                  cf_ffn_channels: Optional[int]=2,
#                  ffn_dropout: Optional[float]=0.0,
#                  dropout:Optional[float]=0.1,
#                  dilation: Optional[int] = 1):
#
#         super(cf_meta_v3, self).__init__()
#
#         # structure parameters for computing madds
#         self.n_blocks = n_blocks
#         self.in_channels = in_channels
#         self.cf_s_channels = cf_s_channels
#         self.cf_ffn_channels = cf_ffn_channels
#         self.big_kernel_size = big_kernel_size
#         self.meta_encoder = meta_encoder
#         self.fusion_method = fusion_method
#         self.bias = bias
#         self.use_pe = use_pe
#
#         # **************************************************************************************dim adaptive, local rep
#         self.dim_adaptive=nn.Sequential(
#             ConvLayer(
#                 opts=opts, in_channels=in_channels, out_channels=cf_s_channels,
#                 kernel_size=3, stride=1, use_norm=True, use_act=True, dilation=dilation, groups=min(cf_s_channels, in_channels)
#             ),
#             ConvLayer(
#                 opts=opts, in_channels=cf_s_channels, out_channels=cf_s_channels,
#                 kernel_size=1, stride=1, use_norm=False, use_act=False
#             )
#         )
#
#
#         # **************************************************************************************cnn-former, global rep
#         spatial_global_conv = []
#
#         if meta_encoder == 'big_kernel':
#             for block_idx in range(n_blocks):
#                 spatial_global_conv.append(cf_meta_bk_encoder(dim=cf_s_channels // 2,
#                                                               big_kernel_size=big_kernel_size,
#                                                               mid_mix=mid_mix,
#                                                               bias=bias,
#                                                               ffn_dim=cf_ffn_channels,
#                                                               ffn_dropout=ffn_dropout,
#                                                               dropout=dropout))
#         elif meta_encoder == 'global_conv':
#             spatial_global_conv.append(cf_meta_gc_encoder(dim=cf_s_channels//2,
#                                                           meta_kernel_size=meta_kernel_size,
#                                                           instance_kernel_method=instance_kernel_method,
#                                                           use_pe=use_pe,
#                                                           mid_mix=mid_mix,
#                                                           bias=bias,
#                                                           ffn_dim=cf_ffn_channels,
#                                                           ffn_dropout=ffn_dropout,
#                                                           dropout=dropout
#                                                           ))
#             for block_idx in range(1, n_blocks):
#                 spatial_global_conv.append(cf_meta_gc_encoder(dim=cf_s_channels // 2,
#                                                               meta_kernel_size=meta_kernel_size,
#                                                               instance_kernel_method=instance_kernel_method,
#                                                               use_pe=False,
#                                                               mid_mix=mid_mix,
#                                                               bias=bias,
#                                                               ffn_dim=cf_ffn_channels,
#                                                               ffn_dropout=ffn_dropout,
#                                                               dropout=dropout
#                                                               ))
#         elif meta_encoder == 'global_conv_ca':
#             spatial_global_conv.append(cf_meta_gc_encoder_ca(dim=cf_s_channels//2,
#                                                              meta_kernel_size=meta_kernel_size,
#                                                              instance_kernel_method=instance_kernel_method,
#                                                              use_pe=use_pe,
#                                                              mid_mix=mid_mix,
#                                                              bias=bias,
#                                                              ffn_dim=cf_ffn_channels,
#                                                              ffn_dropout=ffn_dropout,
#                                                              dropout=dropout
#                                                              ))
#             for block_idx in range(1, n_blocks):
#                 spatial_global_conv.append(cf_meta_gc_encoder_ca(dim=cf_s_channels // 2,
#                                                                  meta_kernel_size=meta_kernel_size,
#                                                                  instance_kernel_method=instance_kernel_method,
#                                                                  use_pe=False,
#                                                                  mid_mix=mid_mix,
#                                                                  bias=bias,
#                                                                  ffn_dim=cf_ffn_channels,
#                                                                  ffn_dropout=ffn_dropout,
#                                                                  dropout=dropout
#                                                                  ))
#
#         elif meta_encoder == 'global_conv_sca':
#             spatial_global_conv.append(cf_meta_gc_encoder_sca(dim=cf_s_channels//2,
#                                                              meta_kernel_size=meta_kernel_size,
#                                                              instance_kernel_method=instance_kernel_method,
#                                                              use_pe=use_pe,
#                                                              mid_mix=mid_mix,
#                                                              bias=bias,
#                                                              ffn_dim=cf_ffn_channels,
#                                                              ffn_dropout=ffn_dropout,
#                                                              dropout=dropout
#                                                              ))
#             for block_idx in range(1, n_blocks):
#                 spatial_global_conv.append(cf_meta_gc_encoder_sca(dim=cf_s_channels // 2,
#                                                                  meta_kernel_size=meta_kernel_size,
#                                                                  instance_kernel_method=instance_kernel_method,
#                                                                  use_pe=False,
#                                                                  mid_mix=mid_mix,
#                                                                  bias=bias,
#                                                                  ffn_dim=cf_ffn_channels,
#                                                                  ffn_dropout=ffn_dropout,
#                                                                  dropout=dropout
#                                                                  ))
#
#         elif meta_encoder == 'global_conv_scag':
#             spatial_global_conv.append(cf_meta_gc_encoder_scag(dim=cf_s_channels//2,
#                                                              meta_kernel_size=meta_kernel_size,
#                                                              instance_kernel_method=instance_kernel_method,
#                                                              use_pe=use_pe,
#                                                              mid_mix=mid_mix,
#                                                              bias=bias,
#                                                              ffn_dim=cf_ffn_channels,
#                                                              ffn_dropout=ffn_dropout,
#                                                              dropout=dropout
#                                                              ))
#             for block_idx in range(1, n_blocks):
#                 spatial_global_conv.append(cf_meta_gc_encoder_scag(dim=cf_s_channels // 2,
#                                                                  meta_kernel_size=meta_kernel_size,
#                                                                  instance_kernel_method=instance_kernel_method,
#                                                                  use_pe=False,
#                                                                  mid_mix=mid_mix,
#                                                                  bias=bias,
#                                                                  ffn_dim=cf_ffn_channels,
#                                                                  ffn_dropout=ffn_dropout,
#                                                                  dropout=dropout
#                                                                  ))
#
#
#         elif meta_encoder == 'global_conv_dk':
#             spatial_global_conv.append(cf_meta_gc_encoder_dk(dim=cf_s_channels//2,
#                                                              meta_kernel_size=meta_kernel_size,
#                                                              instance_kernel_method=instance_kernel_method,
#                                                              use_pe=use_pe,
#                                                              mid_mix=mid_mix,
#                                                              bias=bias,
#                                                              ffn_dim=cf_ffn_channels,
#                                                              ffn_dropout=ffn_dropout,
#                                                              dropout=dropout
#                                                              ))
#             for block_idx in range(1, n_blocks):
#                 spatial_global_conv.append(cf_meta_gc_encoder_dk(dim=cf_s_channels // 2,
#                                                                  meta_kernel_size=meta_kernel_size,
#                                                                  instance_kernel_method=instance_kernel_method,
#                                                                  use_pe=False,
#                                                                  mid_mix=mid_mix,
#                                                                  bias=bias,
#                                                                  ffn_dim=cf_ffn_channels,
#                                                                  ffn_dropout=ffn_dropout,
#                                                                  dropout=dropout
#                                                                  ))
#
#
#         spatial_global_conv.append(nn.BatchNorm2d(num_features=cf_s_channels))
#         self.spatial_global_rep = nn.Sequential(*spatial_global_conv)
#
#
#         # **************************************************************************************************fusion layer
#         if fusion_method == 'add':
#             self.channel_adaptive = ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels, groups=min(cf_s_channels, in_channels),
#                                               kernel_size=1, stride=1, use_norm=True, use_act=True)
#             self.local_global_fusion = ConvLayer(opts=opts, in_channels=in_channels, out_channels=in_channels,
#                                                  kernel_size=3, stride=1, use_norm=True, use_act=True)
#
#
#         elif fusion_method == 'concat':
#             self.channel_adaptive = ConvLayer(opts=opts, in_channels=cf_s_channels, out_channels=in_channels, groups=min(cf_s_channels, in_channels),
#                                               kernel_size=1, stride=1, use_norm=True, use_act=True)
#             self.local_global_fusion = ConvLayer(opts=opts, in_channels=2*in_channels, out_channels=in_channels,
#                                                  kernel_size=3, stride=1, use_norm=True, use_act=True)
#
#         self.ca = CA_layer(in_channels)
#
#         self.fusion_method = fusion_method
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         res = x
#
#         x = self.dim_adaptive(x)
#         x = self.spatial_global_rep(x)
#
#         x = self.channel_adaptive(x)
#
#         if self.fusion_method == 'add':
#             x = self.local_global_fusion(res + x)
#         elif self.fusion_method == 'concat':
#             x = self.local_global_fusion(torch.cat((x, res), dim=1))
#
#         x = self.ca(x)
#
#         return x
#
#     def profile_module(self, input: Tensor) -> (Tensor, float, float):
#         params = macs = 0.0
#         b, c, h, w = input.shape
#
#         # local rep
#         out_da, p_da, m_da = module_profile(module=self.dim_adaptive, x=input)
#         params, macs = params+p_da, macs+m_da
#
#         # global rep
#         p_global_rep = sum([p.numel() for p in self.spatial_global_rep.parameters()])
#
#         if self.meta_encoder == 'big_kernel':
#             m_global_s = (self.big_kernel_size * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
#                         self.cf_s_channels + self.cf_ffn_channels) * (h * w)
#
#             m_global_rep = (m_global_s + m_global_ffn) * self.n_blocks
#
#         elif self.meta_encoder == 'global_conv':
#             m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
#                         self.cf_s_channels + self.cf_ffn_channels) * (h * w)
#
#             m_global_rep = (m_global_s + m_global_ffn) * self.n_blocks
#             if self.use_pe:
#                 m_global_rep += c * h * w
#
#         elif self.meta_encoder == 'global_conv_ca':
#             m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
#                         self.cf_s_channels + self.cf_ffn_channels) * (h * w)
#             m_ca = self.cf_ffn_channels*(self.cf_ffn_channels//16) * (h*w) * 2
#
#             m_global_rep = (m_global_s + m_global_ffn + m_ca) * self.n_blocks
#             if self.use_pe:
#                 m_global_rep += c * h * w
#
#         elif self.meta_encoder == 'global_conv_sca':
#             m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
#                         self.cf_s_channels + self.cf_ffn_channels) * (h * w)
#             m_ca = self.cf_ffn_channels*(self.cf_ffn_channels//16) * (h*w) * 2
#
#             m_global_rep = (m_global_s + m_global_ffn + m_ca) * self.n_blocks
#             if self.use_pe:
#                 m_global_rep += c * h * w
#
#
#         elif self.meta_encoder == 'global_conv_scag':
#             m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 / 4 + (
#                     self.cf_s_channels + self.cf_ffn_channels) * (h * w) / 4
#             m_ca = self.cf_ffn_channels * (self.cf_ffn_channels // 16) * (h * w) * 2
#
#             m_global_rep = (m_global_s + m_global_ffn + m_ca) * self.n_blocks
#             if self.use_pe:
#                 m_global_rep += c * h * w
#
#
#         elif self.meta_encoder == 'global_conv_dk':
#             m_global_s = (h * 1.0) * (1.0 * self.cf_s_channels) * (h * w) * 2
#             if self.bias:
#                 m_global_s += self.cf_s_channels * (h * w) * 2
#             m_global_ffn = self.cf_s_channels * self.cf_ffn_channels * (h * w) * 2 + (
#                         self.cf_s_channels + self.cf_ffn_channels) * (h * w)
#             m_generate_k = (3*1) * (self.cf_s_channels//2*1) * (h*1)*4
#
#             m_global_rep = (m_global_s + m_global_ffn + m_generate_k) * self.n_blocks
#             if self.use_pe:
#                 m_global_rep += c * h * w
#
#
#         params, macs = params + p_global_rep, macs + m_global_rep
#
#
#         # fusion part
#         p_c_adaptive = sum([p.numel() for p in self.channel_adaptive.parameters()])
#         m_c_adaptive = self.cf_s_channels * self.in_channels * (w * h) / min(self.cf_s_channels, self.in_channels)
#
#         p_l_g_fusion = sum([p.numel() for p in self.local_global_fusion.parameters()])
#         if self.fusion_method == 'add':
#             m_l_g_fusion = 3*3*self.in_channels * self.in_channels * (w * h)
#         elif self.fusion_method == 'concat':
#             m_l_g_fusion = 3*3*2*self.in_channels * self.in_channels * (w * h)
#
#         params, macs = params+p_c_adaptive+p_l_g_fusion, macs + m_c_adaptive + m_l_g_fusion
#
#         return input, params, macs

