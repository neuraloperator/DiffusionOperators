import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from typing import Tuple
import glob
import numpy as np

from .util.setup_logger import get_logger
logger = get_logger(__name__)

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

        X_buf = torch.zeros(ntrain, 2, 128, 128).float()
        for i, f in enumerate(files):
            dtype = np.float32
            with open(f, 'rb') as fn:
                load_arr = np.frombuffer(fn.read(), dtype=dtype)
                img = np.array(load_arr.reshape((128, 128, -1)))
            phi = np.angle(img[:,:,0] + img[:,:,1]*1j)            
            X_buf[i,0,:,:] = torch.cos(torch.tensor(phi))
            X_buf[i,1,:,:] = torch.sin(torch.tensor(phi))
        X_buf = X_buf.transpose(1, 2).transpose(2, 3)

        self.x_train = X_buf
        self.transform = transform

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

    def __extra_repr__(self):
        return "shape={}, min={}, max={}".format(len(self.x_train), 
                                                 self.x_train.min(), 
                                                 self.x_train.max())
