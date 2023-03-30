import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce

import copy
import math

# from neuralop.models.tfno import FactorizedFNO1d, FactorizedFNO2d
from time_embedding import TimestepEmbedding

class FNO(nn.Module):
    def __init__(self, s=64, modes=32, width=64, embed_dim=512):
        super().__init__()

        self.fno = FactorizedFNO1d(in_channels=3, width=width, modes_height=modes, factorization=None)
        self.time = TimestepEmbedding(embed_dim, s, s)

        pos_embed = torch.linspace(0, 1, s+1)[0:-1].view(1,1,-1)
        self.register_buffer('pos_embed', pos_embed)
    
    def forward(self, x, sigma):
        bsize = x.size(0)

        x = x.unsqueeze(1)
        time_embed = self.time(sigma).view(1,1,-1).repeat(bsize,1,1)
        pos_embed = self.pos_embed.repeat(bsize,1,1)

        x = torch.cat((x, pos_embed, time_embed), 1)


        return self.fno(x).squeeze(1)

class InterpModule(nn.Module):
    def __init__(self, module, s):
        super().__init__()

        self.module = module
        self.s = s
    
    def forward(self, x):
        x = self.module(x).unsqueeze(1)

        return F.interpolate(x, size=self.s, mode='linear')

class InterpModel(nn.Module):
    def __init__(self, model, s):
        super().__init__()

        self.model = model

        pos_embed = torch.linspace(0, 1, s+1)[0:-1].view(1,1,-1)
        self.model.pos_embed = pos_embed

        self.old_time = copy.deepcopy(self.model.time)
        self.model.time = InterpModule(self.old_time, s)
    
    def forward(self, x, sigma):
        return self.model(x, sigma)


class FNO2d(nn.Module):
    def __init__(self, s=64, modes=32, width=64, out_channels = 1, in_channels = 1, embed_dim=512):
        super().__init__()

        self.s = s

        self.fno = FactorizedFNO2d(in_channels=in_channels+3, out_channels=out_channels, width=width, modes_height=modes, modes_width=modes, factorization=None)
        self.time = TimestepEmbedding(embed_dim, 2*s, s**2, pos_dim=1)

        t = torch.linspace(0, 1, s+1)[0:-1]
        X, Y = torch.meshgrid(t, t, indexing='ij') 

        self.register_buffer('pos_embed_x', X)
        self.register_buffer('pos_embed_y', Y)
    
    def forward(self, x, sigma):
        bsize = x.size(0)

       # x = x.unsqueeze(1)
        x = torch.permute(x, (0, 3, 1, 2))
        time_embed = self.time(sigma).view(1,1,self.s, self.s).repeat(bsize,1,1,1)
        pos_embed_x = self.pos_embed_x.repeat(bsize,1,1,1)
        pos_embed_y = self.pos_embed_y.repeat(bsize,1,1,1)

        x = torch.cat((x, pos_embed_x, pos_embed_y, time_embed), 1)


        return torch.permute(self.fno(x).squeeze(1),(0, 2, 3, 1))






class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2, modes1 = None, modes2 = None, fmult=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        if fmult is not None:
            self.modes1 = int(self.modes1 * fmult)
            self.modes2 = int(self.modes2 * fmult)
        self.fmult = fmult
        
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        dim1, dim2 = self.dim1, self.dim2
        #if x.size(-2) != dim1 or x.size(-1) != dim2:
        #    raise ValueError(("x is expected to be of spatial dimension ({},{}) " + \
        #        "but dimension of x given is ({},{}) instead").format(
        #            dim1, dim2, x.size(-2), x.size(-1)))
        
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        #print(x.shape, self.modes1, self.modes2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  dim1, dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(dim1, dim2))
        return x

    def extra_repr(self):
        return "in_channels={}, out_channels={}, modes1={}, modes2={}, fmult={}".format(
            self.in_channels, self.out_channels, self.modes1, self.modes2, self.fmult
        )


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel, dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        dim1, dim2 = self.dim1, self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2, fmult, groups=8):
        super().__init__()
        self.conv0 = SpectralConv2d(in_channels, out_channels, dim1, dim2, fmult=fmult)
        self.w0 = pointwise_op(in_channels, out_channels, dim1, dim2) 
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x, scale_shift=None):
        x1_c0 = self.conv0(x)
        x2_c0 = self.w0(x)
        x_c0 = x1_c0 + x2_c0
        # then apply norm
        x_c0 = self.norm(x_c0)
        # then scale and shift if exists
        if scale_shift is not None:
            scale, shift = scale_shift
            x_c0 = x_c0 * (scale + 1) + shift
        # then relu
        return F.gelu(x_c0)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dim1, dim2, fmult, groups=8):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, 
                            dim1, dim2,
                            fmult=fmult,
                            groups=groups)
        self.block2 = Block(out_channels, out_channels, 
                            dim1, dim2,
                            fmult=fmult,
                            groups=groups) 
        self.res_conv = SpectralConv2d(in_channels, out_channels,
                                       dim1, dim2,
                                       fmult=fmult)

        self.mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_dim, out_channels * 2)
        )
        
    def forward(self, x, time_emb):
        
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

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
    def __init__(self, in_d_co_domain, d_co_domain, s , pad = 0, fmult=1.0, mult_dims=[1,2,4,4], factor=None, embed_dim=512):
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
        sdim = s + pad

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.block1 = ResnetBlock(
            self.d_co_domain, A[0]*self.d_co_domain, 
            time_dim, 
            sdim, sdim,
            fmult
            #int(sdim*0.75*fmult), int(sdim*0.75*fmult)
        )
        self.block2 = ResnetBlock(
            A[0]*self.d_co_domain, A[1]*self.d_co_domain, 
            time_dim, 
            int(sdim*0.75), int(sdim*0.75),
            fmult,
            #int(sdim/2*fmult), int(sdim/2*fmult)
        )
                                  
        self.block3 = ResnetBlock(
            A[1]*self.d_co_domain, A[2]*self.d_co_domain, 
            time_dim, 
            sdim//2, sdim//2,
            fmult,
            #int(sdim/4*fmult), int(sdim/4*fmult)
        )
                                  
        self.block4 = ResnetBlock(
            A[2]*self.d_co_domain, A[3]*self.d_co_domain, 
            time_dim,
            sdim//4, sdim//4,
            fmult,
            #int(sdim/8*fmult), int(sdim/8*fmult)
        )
                                  
        
        self.inv2_9 = ResnetBlock(
            A[3] * self.d_co_domain, 
            A[2]*self.d_co_domain, 
            time_dim, 
            sdim//2, sdim//2,
            fmult,
            #int(sdim/4*fmult), int(sdim/4*fmult)
        )
        # combine out channels of inv2_9 and block3
        self.inv3 = ResnetBlock(
            (A[2]*2) * self.d_co_domain, 
            A[1]*self.d_co_domain, 
            time_dim, 
            int(sdim*0.75), int(sdim*0.75),
            fmult,
            #int(sdim/2*fmult), int(sdim/2*fmult)
        )
        # combine out channels of inv3 and block2
        self.inv4 = ResnetBlock(
            (A[1]*2) * self.d_co_domain, 
            A[0]*self.d_co_domain, 
            time_dim, 
            sdim, sdim,
            fmult,
            #int(sdim*0.75*fmult), int(sdim*0.75*fmult)
        )
        
        # combine out channels of inv4 and block1
        self.inv5 = ResnetBlock(
            (A[0]*2) * self.d_co_domain, 
            self.d_co_domain, 
            time_dim,
            sdim, sdim,
            fmult,
            #int(sdim*fmult), int(sdim*fmult)
        )

        self.post = ResnetBlock(
            self.d_co_domain*2, 
            self.d_co_domain,
            time_dim,
            sdim, sdim,
            fmult
        )
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
