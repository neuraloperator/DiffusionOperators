import torch
from torch.utils.data import Dataset
import glob
import numpy as np

from setup_logger import get_logger
logger = get_logger(__name__)

class VolcanoDataset(Dataset):

    def __init__(self, root, ntrain=4096, transform=None):

        super().__init__()

        res = 128-8
        files = glob.glob("{}/**/*.int".format(root), recursive=True)[:ntrain]
        if len(files) == 0:
            raise Exception("Cannot find any *.int files here.")
        logger.info("# files detected: {}".format(len(files)))
        if len(files) != ntrain:
            raise ValueError("ntrain=={} but we only detected {} files".\
                format(ntrain, len(files)))

        x_train = torch.zeros(ntrain, res, res, 2).float()
        nline = 128
        nsamp = 128
        for i, f in enumerate(files):
            dtype = np.float32

            with open(f, 'rb') as fn:
                load_arr = np.frombuffer(fn.read(), dtype=dtype)
                img = np.array(load_arr.reshape((nline, nsamp, -1)))

            phi = np.angle(img[:,:,0] + img[:,:,1]*1j)
            x_train[i,:,:,0] = torch.cos(torch.tensor(phi[:res, :res]))
            x_train[i,:,:,1] = torch.sin(torch.tensor(phi[:res, :res]))

        self.x_train = x_train
        self.transform = transform

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
