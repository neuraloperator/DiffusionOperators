import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from typing import Tuple, Dict
import glob
import numpy as np
import os

from .util.setup_logger import get_logger
logger = get_logger(__name__)

from scipy.stats import wasserstein_distance as w_distance

from einops import rearrange



# TODO
# dataset should have .X method, .resolution

"""
class DelegableSubset(Subset):
    def __getattr__(self, name):
        if hasattr(self, name):
            # If Subset has this method already, return it
            return self.__dict__[name]
        elif hasattr(self.dataset, name):
            # If self.dataset has this method, return it
            return getattr(self.dataset, name)
        else:
            raise AttributeError(f"object has no attribute '{name}'")
"""

def circular_var(x: torch.Tensor, dim=None):
    """Extracted from: https://github.com/kazizzad/GANO/blob/main/GANO_volcano.ipynb"""
    #R = torch.sqrt((x.mean(dim=(1,2))**2).sum(dim=1))
    phase = to_phase(x)
    
    C1 = torch.cos(phase).sum(dim=(1,2))
    S1 = torch.sin(phase).sum(dim=(1,2))
    R1 = torch.sqrt(C1**2 + S1**2) / (phase.shape[1]*phase.shape[2])
    return 1 - R1

def circular_skew(x: torch.Tensor):
    """Extracted from: https://github.com/kazizzad/GANO/blob/main/GANO_volcano.ipynb"""
    phase = to_phase(x)
    
    C1 = torch.cos(phase).sum(dim=(1,2))
    S1 = torch.sin(phase).sum(dim=(1,2))
    R1 = torch.sqrt(C1**2 + S1**2) / (phase.shape[1]*phase.shape[2])
    
    C2 = torch.cos(2*phase).sum(dim=(1,2))
    S2 = torch.sin(2*phase).sum(dim=(1,2))
    R2 = torch.sqrt(C2**2 + S2**2) / (phase.shape[1]*phase.shape[2])
    
    T1 = torch.atan2(S1, C1)
    T2 = torch.atan2(S2, C2)

    return R2 * torch.sin(T2 - 2*T1) / (1 - R1)**(3/2)

def to_phase(samples: torch.Tensor):
    assert samples.size(-1) == 2, "Last dim should be 2d"
    phase = torch.atan2(samples[...,1], samples[...,0])
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    return phase

class FunctionDataset(Dataset):
    """
    """
    def __init__(self, transform=None):
        self.transform = transform
    
    @property
    def X(self) -> torch.Tensor:
        raise NotImplementedError("This method should return X, a torch tensor for the dataset")
    
    @property
    def res(self) -> int:
        raise NotImplementedError("This method should return the spatial dimension of the dataset")

    @property
    def n_in(self) -> int:
        return None

    def evaluate(self, samples) -> Dict:
        raise NotImplementedError()

    def _to_channels_first(self, x):
        return rearrange(x, 'h w f -> f h w')

    def _to_channels_last(self, x):
        return rearrange(x, 'f h w -> h w f')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idc):
        if self.transform is None:
            return self.X[idc]
        else:
            # convert back to (h, w, f) format
            return self._to_channels_last(
                self.transform(
                    # convert to (f, h, w) format
                    self._to_channels_first(self.X[idc])
                )
            )

    def postprocess(self, samples: torch.Tensor):
        return samples

    @property
    def postprocess_kwargs(self):
        return {}

SPLIT_ALLOWED = ['train', 'test']

class NavierStokesDataset(FunctionDataset):
    def __init__(self, root: str, split: str = 'train', **kwargs):
        super().__init__(**kwargs)
        if split == 'train':
            dataset = torch.load(
                os.path.join(root, "ns", "nsforcing_128_train.pt")
            )['y']
        elif split == 'test':
            dataset = torch.load(
                os.path.join(root, "ns", "nsforcing_128_test.pt")
            )['y']
        else:
            raise ValueError("'split' must be one of either: {}".format(SPLIT_ALLOWED))
        dataset = dataset.unsqueeze(1)
        self.dataset = rearrange(dataset, 'N f h w -> N h w f')
        
    @property
    def res(self) -> int:
        return self.dataset.size(-1)

    @property
    def n_in(self) -> int:
        return 1

    @property
    def X(self) -> torch.Tensor:
        return self.dataset

class VolcanoDataset(FunctionDataset):

    def __init__(self, root, ntrain=4096, resolution=None, transform=None):

        super().__init__()

        files = glob.glob("{}/volcano/*/*.int".format(root), recursive=True)[:ntrain]
        if len(files) == 0:
            raise Exception("Cannot find any *.int files here.")
        logger.info("# files detected: {}".format(len(files)))
        if len(files) != ntrain:
            raise ValueError("ntrain=={} but we only detected {} files".\
                format(ntrain, len(files)))
        if resolution is None:
            resolution = 128

        actual_res = resolution - 8
        X_buf = torch.zeros(ntrain, 2, 128, 128).float()
        for i, f in enumerate(files):
            dtype = np.float32

            with open(f, 'rb') as fn:
                load_arr = np.frombuffer(fn.read(), dtype=dtype)
                img = np.array(load_arr.reshape((128, 128, -1)))

            phi = np.angle(img[:,:,0] + img[:,:,1]*1j)            
            X_buf[i,0,:,:] = torch.cos(torch.tensor(phi))
            X_buf[i,1,:,:] = torch.sin(torch.tensor(phi))

        X_buf = F.interpolate(X_buf, size=(resolution, resolution),
                              mode='bilinear')
        X_buf = X_buf[:, :, :actual_res, :actual_res].\
            transpose(1, 2).transpose(2, 3)

        self.x_train = X_buf
        self.transform = transform

        self.var_train = circular_var(self.x_train).numpy()
        self.skew_train = circular_skew(self.x_train).numpy()

    def evaluate(self, samples: torch.Tensor) -> Dict:
        """Given a tensor of generated samples, eval dataset-specific metrics"""
        if list(samples.shape)[1:] != list(self.x_train.shape)[1:]:
            raise ValueError("Shape mismatch: samples=(bs,)+{}, dataset=(bs,)+{}".\
                format(samples.shape[1:], self.x_train.shape[1:]))
        skew_generated = circular_skew(samples)
        var_generated = circular_var(samples)
        w_skew = w_distance(self.skew_train, skew_generated)
        w_var = w_distance(self.var_train, var_generated)
        w_total = w_skew + w_var
        metric_vals = {
            "w_skew": w_skew, 
            "w_var": w_var,
            "w_total": w_total
        }
        return metric_vals

    def postprocess(self, samples: torch.Tensor) -> torch.Tensor:
        """Postprocess an image for plotting purposes"""
        return to_phase(samples)

    @property
    def postprocess_kwargs(self):
        """extra kwargs for ax.imshow()"""
        return dict(
            cmap='RdYlBu', 
            vmin = -np.pi, 
            vmax=np.pi,
            extent=[0,1,0,1]
        )

    @property
    def X(self):
        return self.x_train

    @property
    def res(self):
        return self.x_train.size(1)