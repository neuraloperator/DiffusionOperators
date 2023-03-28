import torch

from ncsnv2_models import ncsnv2

from utils import DotDict

config = DotDict()
config.data = DotDict()
config.data.logit_transform = False
config.data.rescaled = False
config.model = DotDict()
config.model.ngf = 128
config.model.sigma_dist = 'geometric'
config.model.sigma_begin = 30
config.model.sigma_end = 0.01
config.model.num_classes = 100
config.model.nonlinearity = 'elu'
config.model.normalization = "InstanceNorm++"
config.data.channels = 2
config.data.image_size = 128

config.device = 0 

model = ncsnv2.NCSNv2(config)
model.to(config.device)

xfake = torch.randn((16,2,128,128)).to(config.device)
ys = torch.arange(0, 16).long().to(config.device)
print(model(xfake, ys).shape)