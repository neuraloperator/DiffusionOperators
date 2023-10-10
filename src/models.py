import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List

from neuralop.models import UNO

from .util.setup_logger import get_logger
logger = get_logger(__name__)

from .util.utils import count_params

from typing import Union, Tuple

from neuralop.layers.resample import resample

class UNO_MyClass(UNO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final = nn.Conv2d(self.uno_out_channels[-1], self.out_channels, 1)

    def forward(self, x, **kwargs):
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
                
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)
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
            norm=norm,
            uno_n_modes=n_modes,
            horizontal_skips_map=horizontal_skips_map,
            uno_scalings=uno_scalings,
            factorization=factorization,
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
        result = self.uno(x) / sigmas
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