import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from typing import List

from neuralop.models import UNO

from .util.setup_logger import get_logger
logger = get_logger(__name__)

from .util.utils import count_params

from typing import Union, Tuple

from neuralop.layers.resample import resample
from neuralop.layers.fno_block import FNOBlocks 

def get_sinusoidal_positional_embedding(
    timesteps: torch.FloatTensor,
    embedding_dim: int,
    scale: float = 10000.,
):
    """
    Copied and modified from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90
    From Fairseq in https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    #assert len(timesteps.size()) == 1  # and timesteps.dtype == tf.int32
    assert len(timesteps.size()) == 1 or len(timesteps.size()) == 2  # and timesteps.dtype == tf.int32
    if len(timesteps.size()) == 1:
        batch_size = timesteps.size(0)
        index_dim = 1
    else:
        batch_size, index_dim = timesteps.size()
        timesteps = timesteps.view(batch_size*index_dim)
    timesteps = timesteps.to(torch.get_default_dtype())#float()
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(scale) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    # emb = torch.arange(num_embeddings, dtype=torch.float, device=device)[:, None] * emb[None, :]
    emb = timesteps[..., None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad to the last dimension
        # emb = torch.cat([emb, torch.zeros(num_embeddings, 1, device=device)], dim=1)
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [batch_size*index_dim, embedding_dim]
    return emb.view(batch_size, index_dim*embedding_dim)

class FNOBlocks_MyClass(FNOBlocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t_mlps = []
        for _ in range(self.n_layers*self.n_norms):
            t_mlps.append(nn.Sequential(
                nn.Linear(256, self.out_channels),
                nn.GELU(),
                nn.Linear(self.out_channels, self.out_channels*2)
            ))
        self.t_mlps = nn.ModuleList(t_mlps)        

    def forward(self, x, index=0, output_shape=None, t_emb=None):
        """Just override forward, assume we are doing forward_with_postactivation"""
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)
        t_scale, t_shift = self.t_mlps[self.n_norms*index](t_emb).chunk(2, dim=1)
        x_fno = x_fno*t_scale.unsqueeze(-1).unsqueeze(-1) + \
            t_shift.unsqueeze(-1).unsqueeze(-1)

        x = x_fno + x_skip_fno

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

class UNO_MyClass(UNO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final = nn.Conv2d(self.uno_out_channels[-1], self.out_channels, 1)

    def forward(self, x, t_emb=None):
        x = self.lifting(x)
        #print("lift:", x.shape)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
            #print("pad: ", x.shape)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                #output_scaling_factors = [
                #    m / n for (m, n) in zip(x.shape, skip_val.shape)
                #]
                #output_scaling_factors = output_scaling_factors[-1 * self.n_dim :]
                #t = resample(
                #    skip_val, output_scaling_factors, list(range(-self.n_dim, 0))
                #)
                #print("    ", x.shape, ">>", skip_val.shape)
                x = torch.cat([x, skip_val], dim=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
                #print("cur_output:", cur_output)
                
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output, t_emb=t_emb)
            #print("{}:".format(layer_idx), x.shape)

            if layer_idx in self.horizontal_skips_map.values():
                #skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)
                skip_outputs[layer_idx] = x

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
            #print("unpad:", x.shape)

        #x = self.projection(x)
        x = self.final(x)
        #print(x.shape)
        
        return x

class UNO_Diffusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_width: int,
                 spatial_dim: int,
                 mult_dims: List[float],
                 npad: int, 
                 fmult: float,
                 rank: float = 1.0,
                 norm: Union[str, None] = None,
                 factorization: str = None,):
        super().__init__()
        
        # using [1,2,4,4] as mults
        bw = base_width                         # base width
        sdim = spatial_dim                      # spatial dim

        # Compared to the old code, we were also going from
        # 128 -> 96 -> 64 but maybe it's faster to just downscale
        # with powers of two.
        uno_scalings = [
            1.0,    # 0, 128px
            0.75,   # 1, 96px
            0.67,   # 2, 64px
            
            0.5,    # 3, 32px
            
            2.0,    # 4, 64px
            1.5,    # 5, 96px
            1.33,   # 6, 128px
            1.0,    # 7, 128px
        ]
        horizontal_skips_map = {
            7: 0,
            6: 1,
            5: 2,
        }
        
        # The number of fourier modes is just the resolution at that
        # layer multiplied by `fmult`.
        n_modes = []
        _curr_res = [sdim, sdim]
        for scale_factor in uno_scalings:
            _curr_res[0] = _curr_res[0]*scale_factor
            _curr_res[1] = _curr_res[1]*scale_factor
            max_modes = int((_curr_res[0]*_curr_res[1]) // 2)
            max_modes_per_res = int(np.sqrt(max_modes))
            # n modes is basically res//2 (well, +1 as well)
            # so do (res//2) * fmult
            n_modes.append(
                (int(max_modes_per_res*fmult), int(max_modes_per_res*fmult))
            )
            logger.debug("{}x{}: max mode per res: {}, # retained per res: {}".format(
                int(_curr_res[0]), int(_curr_res[1]),
                max_modes_per_res,
                int(max_modes_per_res*fmult)
            ))

        #logger.debug("fmult={}, n_modes={}".format(
        #    fmult, n_modes
        #))

        logger.debug("norm: {}".format(norm))

        pad_factor = (float(npad)/2) / sdim
        self.uno = UNO_MyClass(
            in_channels=in_channels+2,  # +2 because of grid we pass
            out_channels=out_channels,
            hidden_channels=bw,         # lift input to this no. channels   
            n_layers=len(uno_scalings), # number of fourier layers
            uno_out_channels=[
                bw*mult_dims[0], 
                bw*mult_dims[1],
                bw*mult_dims[2],
                bw*mult_dims[3],
                #
                bw*mult_dims[3],
                bw*mult_dims[2],
                bw*mult_dims[1],
                bw*mult_dims[0],
            ],
            operator_block=FNOBlocks_MyClass,
            norm=norm,
            uno_n_modes=n_modes,
            horizontal_skips_map=horizontal_skips_map,
            uno_scalings=uno_scalings,
            factorization=factorization,
            implementation='factorized',
            rank=rank,
            domain_padding_mode='symmetric',
            domain_padding=pad_factor,
        )
        #print(self.uno)

    def get_grid(self, shape: Tuple[int]):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)
        
    def forward(self, x: torch.FloatTensor, sigmas: torch.FloatTensor):
        """As per the paper 'Improved techniques for training SBGMs',
          define s(x; sigma_t) = f(x) / sigma_t.
        """
        
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.swapaxes(-1,-2).swapaxes(-2,-3)
        
        t_emb = get_sinusoidal_positional_embedding(sigmas.flatten(), 256)
        result = self.uno(x, t_emb=t_emb) / sigmas
        return result.swapaxes(1,2).swapaxes(2,3)

if __name__ == '__main__':

    #uno = UNO_Diffusion(2, 1, 32, 128, 0.25)
    #print(count_params(uno))

    xfake = torch.randn((4, 2, 128, 128))

    from neuralop.layers.padding import DomainPadding

    pad = DomainPadding(
        domain_padding=(4./128.),
        padding_mode='symmetric',
        output_scaling_factor=1
    )

    print(pad.pad(xfake).shape)