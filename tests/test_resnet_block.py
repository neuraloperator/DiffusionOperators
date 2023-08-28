import torch

from src.models import ResnetBlock
from src.util.utils import count_params

def test_resnet_block_spatial_dims():
    # in_dim and out_dim are currently required for this class.
    # they control the number of fourier modes used in the
    # convolution.
    block = ResnetBlock(
        in_channels=16, 
        out_channels=32, 
        in_dim=128, 
        out_dim=64, 
        time_dim=10
    )
    xfake = torch.randn((8, 16, 128, 128))
    out = block(xfake, None, None)
    # (8, 32, 64, 64)
    print("block1 on 128x128: {}".format(out.shape))

    xfake2 = torch.randn((8, 16, 256, 256))
    out2 = block(xfake2, None, None)
    print("block1 on 256x256: {}".format(out2.shape))
    #print(count_params(block2))

    # this will use more parameters because the number of
    # modes is determined by fmult*in_dim
    block2 = ResnetBlock(
        in_channels=16, 
        out_channels=32, 
        in_dim=128*2, 
        out_dim=64*2, 
        time_dim=10
    )   

    print(count_params(block), count_params(block2))

if __name__ == '__main__':
    test_resnet_block_spatial_dims()