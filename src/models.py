import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models import UNO

from .util.setup_logger import get_logger
logger = get_logger(__name__)

from .util.utils import count_params

class UNO_Diffusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_width: int,
                 spatial_dim: int,
                 npad: int, 
                 fmult: float,
                 rank: float = 1.0,
                 factorization: str = None,):
        super().__init__()
        
        # using [1,2,4,4] as mults
        bw = base_width                         # base width
        sdim = spatial_dim                      # spatial dim
        # Compared to the old code, we were also going from
        # 128 -> 96 -> 64 but maybe it's faster to just downscale
        # with powers of two.
        uno_scalings = [
            (0.5, 0.5),   # e.g. (128 -> 64)
            (0.5, 0.5),   # e.g. (64 -> 32)    
            (1.0, 1.0),   # e.g. (32 -> 32)
            (2.0, 2.0),   # e.g. (32 -> 64)
            (2.0, 2.0),   # e.g. (64 -> 128),
        ]
        # The number of fourier modes is just the resolution at that
        # layer multiplied by `fmult`.
        n_modes = []
        _curr_res = [sdim, sdim]
        for tp in uno_scalings:
            _curr_res[0] = _curr_res[0]*tp[0]*fmult
            _curr_res[1] = _curr_res[1]*tp[1]*fmult
            n_modes.append((int(_curr_res[0]), int(_curr_res[1])))

        pad_factor = (float(npad)/2) / sdim
        self.uno = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=bw,         # lift input to this no. channels   
            n_layers=len(uno_scalings), # number of fourier layers
            uno_out_channels=[
                bw*2,
                bw*4,
                #
                bw*4,
                #
                bw*4,
                bw*2,
            ],
            uno_n_modes=n_modes,
            uno_scalings=uno_scalings,
            factorization=factorization,
            rank=rank,
            domain_padding_mode='symmetric',
            domain_padding=pad_factor,
        )
    def forward(self, x: torch.FloatTensor, sigmas: torch.FloatTensor):
        """As per the paper 'Improved techniques for training SBGMs',
          define s(x; sigma_t) = f(x) / sigma_t.
        """        
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