import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from functools import partial

import copy
import math

# from neuralop.models.tfno import FactorizedFNO1d, FactorizedFNO2d
from time_embedding import TimestepEmbedding

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels,groups=8):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv0(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, mode='ds', groups=8):
        super().__init__()
        assert mode in ['ds', 'us', None]
        self.block1 = Block(in_channels, out_channels, 
                            groups=groups)
        self.block2 = Block(out_channels, out_channels,
                            groups=groups)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        

        self.mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_dim, out_channels * 2)
        )
        if mode == 'ds':
            self.post = Downsample(out_channels, out_channels)
        elif mode == 'us':
            self.post = Upsample(out_channels, out_channels)
        else:
            self.post = nn.Identity()
        
    def forward(self, x, time_emb):
        
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return self.post( h + self.res_conv(x) )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        
class UNO(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, s , pad = 0, mult_dims=[1,2,4,4], factor=None, embed_dim=512):
        super(UNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.s = s
        #self.time = TimestepEmbedding(embed_dim, 2*self.s, self.s**2, pos_dim=1)
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        #self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        time_dim = d_co_domain*4

        self.init_conv = nn.Conv2d(in_d_co_domain, d_co_domain, 1, padding=0)

        #self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)


        A = mult_dims

        # Currently assumes 128px input.

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.block1 = ResnetBlock(self.d_co_domain, A[0]*self.d_co_domain, time_dim, mode='ds')
        self.block2 = ResnetBlock(A[0]*self.d_co_domain, A[1]*self.d_co_domain, time_dim, mode='ds')
        self.block3 = ResnetBlock(A[1]*self.d_co_domain, A[2]*self.d_co_domain, time_dim, mode='ds')
        self.block4 = ResnetBlock(A[2]*self.d_co_domain, A[3]*self.d_co_domain, time_dim, mode='ds')
        
        self.inv2_9 = ResnetBlock(A[3] * self.d_co_domain, A[2]*self.d_co_domain, time_dim, mode='us')
        # combine out channels of inv2_9 and block3
        self.inv3 = ResnetBlock( (A[2]*2) * self.d_co_domain, A[1]*self.d_co_domain, time_dim, mode='us')
        # combine out channels of inv3 and block2
        self.inv4 = ResnetBlock( (A[1]*2) * self.d_co_domain, A[0]*self.d_co_domain, time_dim, mode='us')
        # combine out channels of inv4 and block1
        self.inv5 = ResnetBlock( (A[0]*2) * self.d_co_domain, self.d_co_domain, time_dim, mode='us') # will be reshaped

        self.post = ResnetBlock(self.d_co_domain*2, self.d_co_domain,
                                time_dim,
                                mode=None)
        self.final_conv = nn.Conv2d(self.d_co_domain, 2, 1, padding=0)
        
        #self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        #self.fc2 = nn.Linear(4*self.d_co_domain, 2)

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor, sigmas: torch.FloatTensor):
        """
        Args:
          x: float tensor of shape (bs, res, res, 2)
          t: long tensor of shape (bs,)
          sigmas: float tensor of shape (bs, 1, 1, 1)

        Notes:

        `x` is preprocessed so that it is transformed into the
          shape (bs, res, res, 5), where 2 comes from the grid
          and 1 comes from the time embedding.

        `t` is preprocessed into a time embedding vector and then
          passed into each residual block to be transformed into
          a scale and shift parameter.

        `sigmas` is used to divide the output by the noise scale, as
        per the suggestion in "Improved techniques for training SBGMs".
        """

        # have a different time embedding per minibatch
        # + as suggested in improved techniques paper, just redefine
        # s(x,sigma) = s(x) / sigma instead.
        
        bsize = x.size(0)
        grid = self.get_grid(x.shape, x.device)
        # print('time_size',self.time(sigma).view(1,self.s, self.s,1).size())
        #time_embed = self.time(sigma).view(1,self.s, self.s,1).repeat(bsize,1,1,1)

        t_emb = self.time_mlp(t)

        x = torch.cat((x, grid), dim=-1)

        x_fc0 = x.permute(0, 3, 1, 2)
        x_fc0 = self.init_conv(x_fc0)

        #x_fc0 = self.fc0(x)
        #x_fc0 = F.gelu(x_fc0)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        #print("x_Fc0: ", x_fc0.shape)
        
        x_c0 = self.block1(x_fc0, t_emb)
        x_c1 = self.block2(x_c0, t_emb)
        x_c2 = self.block3(x_c1, t_emb)
        x_c2_1 = self.block4(x_c2, t_emb)
        
        x_c2_9 = self.inv2_9(x_c2_1, t_emb)
        x_c2_9 = torch.cat((x_c2_9, x_c2), dim=1)

        #print("-----------")

        x_c3 = self.inv3(x_c2_9, t_emb)
        x_c3 = torch.cat((x_c3, x_c1), dim=1)

        x_c4 = self.inv4(x_c3, t_emb)
        x_c4 = torch.cat((x_c4, x_c0), dim=1)

        x_c5 = self.inv5(x_c4, t_emb)
        x_c5 = torch.cat((x_c5, x_fc0), dim=1)
        
        x_c6 = self.post(x_c5, t_emb)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = self.final_conv(x_c6)

        x_out = x_c6.permute(0, 2, 3, 1)        
        
        return x_out / sigmas
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
