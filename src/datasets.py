import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from typing import Tuple, Dict
import glob
import numpy as np

from .util.setup_logger import get_logger
logger = get_logger(__name__)

from scipy.stats import wasserstein_distance as w_distance



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
    This abstraction was created for the following reasons:
    - To make it clear that the original dataset gets cropped.
    - To provide convenience functions for cropping the dataset.
    - To record the 'amount' of the image cropped out (the 'padding')
      as well as the actual resolution of the image.
    """
    
    @property
    def X(self) -> torch.Tensor:
        raise NotImplementedError("This method should return X, a torch tensor for the dataset")
    
    @property
    def res(self) -> int:
        raise NotImplementedError("This method should return the spatial dimension of the dataset")

    def evaluate(self, samples) -> Dict:
        raise NotImplementedError()

class VolcanoDataset(FunctionDataset):
    """
    The resolution of this dataset is 120x120px, obtained by loading in
      the original 128x128 data and cropping out a 120x120 region from 
      the top left corner of the image.

    Notes
    -----

    (Chris B): The cropping decision is weird, why not just center
      crop from the middle, i.e. crop by npad//2 for each edge???
      If we want to emulate what a padded conv does then we should
      be padding each edge as well.
    """

    def __init__(self, root, ntrain=4096, resolution=None, transform=None):

        super().__init__()

        files = glob.glob("{}/**/*.int".format(root), recursive=True)[:ntrain]
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

    def _to_fhw(self, x):
        """Convert x into format (f, h, w)"""
        assert len(x.shape) == 3
        return x.swapaxes(2,1).swapaxes(1,0)

    def _to_hwf(self, x):
        """Convert x into format (h, w, f)"""
        assert len(x.shape) == 3
        return x.swapaxes(0,1).swapaxes(1,2)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idc):
        if self.transform is None:
            return self.x_train[idc]
        else:
            # convert back to (h, w, f) format
            return self._to_hwf(
                self.transform(
                    # convert to (f, h, w) format
                    self._to_fhw(self.x_train[idc])
                )
            )