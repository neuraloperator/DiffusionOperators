import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from run_nerf_helpers import get_embedder

from einops import rearrange, reduce

import copy
import math

from functools import partial
from typing import List, Union

# from neuralop.models.tfno import FactorizedFNO1d, FactorizedFNO2d
from time_embedding import TimestepEmbedding

from neuralop.models.fno_block import FNOBlocks
from neuralop.models.spectral_convolution import FactorizedSpectralConv

from setup_logger import get_logger

logger = get_logger(__name__)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dim: int,
        out_dim: int,
        time_dim: int,
        fmult: float = 1.0,
        groups: int = 8,
        **kwargs
    ):
        """Residual block that encapsulates FNOBlock.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            in_dim (int): expected input spatial dimension.
            out_dim (int): expected output spatial dimension
            time_dim (int): time embedding dimension.
            fmult (float): multiplier to reduce the number of Fourier modes.
              For instance, 1.0 means the maximal possible number will be used,
              and 0.5 means only half will be used. Larger numbers correspond
              to more parameters / memory.
            kwargs: extra args that can be passed to FNOBlocks.
        """

        super().__init__()

        n_modes = (int(in_dim * fmult), int(in_dim * fmult))

        # e.g. 64 / 128 = 0.5 scale factor
        rel_scale_factor = out_dim / in_dim

        if groups == 0:
            # Then no norm is added to the block
            norm_fn = None
        else:
            norm_fn = lambda nc: nn.GroupNorm(num_groups=groups, num_channels=nc)

        self.block1 = FNOBlocks(
            in_channels,
            out_channels,
            n_modes=n_modes,
            output_scaling_factor=(rel_scale_factor, rel_scale_factor),
            SpectralConv=FactorizedSpectralConv,
            use_mlp=False,
            norm=norm_fn,
            **kwargs
        )

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))

    def forward(self, x, time_emb, grid):

        # grid_ds = F.interpolate(grid, size=(x.size(-2), x.size(-1)))
        # x = torch.cat((x, grid_ds), dim=1)

        h = self.block1(x)

        # print(x.shape, "==>", h.shape)

        # h = self.block2(h)
        return h  # + self.res_conv(x)


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


class UNO_Diffusion:
    """Wrapper class which also stores the samplers and noise schedule"""

    def __init__(self, uno, sigma, noise_sampler, init_sampler):
        self.uno = uno
        self.sigma = sigma
        self.noise_sampler = noise_sampler
        self.init_sampler = init_sampler

    def __call__(self, *args, **kwargs):
        return self.uno(*args**kwargs)

    def state_dict(self):
        return self.uno_state_dict()

    def load_state_dict(self, dd):
        return self.uno.load_state_dict(dd)


class UNO(nn.Module):
    """U-Shaped neural operator architecture"""

    def __init__(
        self,
        in_d_co_domain: int,
        d_co_domain: int,
        s: int,
        pad: int = 0,
        groups: int = 0,
        fmult: float = 1.0,
        mult_dims: List[int] = [1, 2, 4, 4],
        num_freqs_input: int = 0,
        **kwargs
    ):
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
        # self.time = TimestepEmbedding(embed_dim, 2*self.s, self.s**2, pos_dim=1)
        self.in_d_co_domain = in_d_co_domain  # input channel
        self.d_co_domain = d_co_domain
        # self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        time_dim = d_co_domain * 4

        # dim_grid = 2
        ##embedder, dim_grid = get_embedder(num_freqs_input, input_dims=in_d_co_domain)
        # self.embedder = embedder
        # self.dim_grid = dim_grid
        # dim_grid = 0

        dim_grid = 0
        self.init_conv = nn.Conv2d(in_d_co_domain + 2, d_co_domain, 1, padding=0)

        # self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        A = mult_dims

        # Currently assumes 128px input.
        sdim = s + pad

        block_class = partial(ResnetBlock, **kwargs)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.block1 = block_class(
            self.d_co_domain + dim_grid,
            A[0] * self.d_co_domain,
            # 128, 128
            sdim,
            sdim,
            time_dim,
            fmult,
            groups=groups
        )
        self.block2 = block_class(
            A[0] * self.d_co_domain + dim_grid,
            A[1] * self.d_co_domain,
            # 128, 96
            sdim,
            int(sdim * 0.75),
            time_dim,
            fmult,
            groups=groups
        )

        self.block3 = block_class(
            A[1] * self.d_co_domain + dim_grid,
            A[2] * self.d_co_domain,
            # 96, 64
            int(sdim * 0.75),
            sdim // 2,
            time_dim,
            fmult,
            groups=groups
        )

        self.block4 = block_class(
            A[2] * self.d_co_domain + dim_grid,
            A[3] * self.d_co_domain,
            # 64, 32
            sdim // 2,
            sdim // 4,
            time_dim,
            fmult,
            groups=groups
        )

        self.inv2_9 = block_class(
            A[3] * self.d_co_domain + dim_grid,
            A[2] * self.d_co_domain,
            # 32, 64
            sdim // 4,
            sdim // 2,
            time_dim,
            fmult,
            groups=groups
        )
        # combine out channels of inv2_9 and block3
        self.inv3 = block_class(
            (A[2] * 2) * self.d_co_domain + dim_grid,
            A[1] * self.d_co_domain,
            # 64, 96
            sdim // 2,
            int(sdim * 0.75),
            time_dim,
            fmult,
            groups=groups
        )
        # combine out channels of inv3 and block2
        self.inv4 = block_class(
            (A[1] * 2) * self.d_co_domain + dim_grid,
            A[0] * self.d_co_domain,
            # 96, 128
            int(sdim * 0.75),
            sdim,
            time_dim,
            fmult,
            groups=groups
            # int(sdim*0.75*fmult), int(sdim*0.75*fmult)
        )

        # combine out channels of inv4 and block1
        self.inv5 = block_class(
            (A[0] * 2) * self.d_co_domain + dim_grid,
            self.d_co_domain,
            sdim,
            sdim,
            time_dim,
            fmult,
            groups=groups
            # int(sdim*fmult), int(sdim*fmult)
        )

        self.final_conv = nn.Conv2d(self.d_co_domain * 2, 2, 1, padding=0)

    def forward(
        self, x: torch.FloatTensor, t: torch.LongTensor, sigmas: torch.FloatTensor
    ):
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
        # grid = self.embedder(grid)

        # print('time_size',self.time(sigma).view(1,self.s, self.s,1).size())
        # time_embed = self.time(sigma).view(1,self.s, self.s,1).repeat(bsize,1,1,1)

        t_emb = self.time_mlp(t)

        x = torch.cat((x, grid), dim=-1)

        x_fc0 = x.permute(0, 3, 1, 2)
        x_fc0 = self.init_conv(x_fc0)

        # grid = grid.permute(0,3,1,2)

        # x_fc0 = self.fc0(x)
        # x_fc0 = F.gelu(x_fc0)
        x_fc0 = F.pad(x_fc0, [0, self.padding, 0, self.padding])

        x_c0 = self.block1(x_fc0, t_emb, grid)
        x_c1 = self.block2(x_c0, t_emb, grid)
        x_c2 = self.block3(x_c1, t_emb, grid)
        x_c2_1 = self.block4(x_c2, t_emb, grid)

        x_c2_9 = self.inv2_9(x_c2_1, t_emb, grid)
        x_c2_9 = torch.cat((x_c2_9, x_c2), dim=1)

        # print("-----------")

        x_c3 = self.inv3(x_c2_9, t_emb, grid)
        x_c3 = torch.cat((x_c3, x_c1), dim=1)

        x_c4 = self.inv4(x_c3, t_emb, grid)
        x_c4 = torch.cat((x_c4, x_c0), dim=1)

        x_c5 = self.inv5(x_c4, t_emb, grid)
        x_c5 = torch.cat((x_c5, x_fc0), dim=1)

        # x_c6 = self.post(x_c5, t_emb)
        x_c6 = x_c5

        if self.padding != 0:
            x_c6 = x_c6[..., : -self.padding, : -self.padding]

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
