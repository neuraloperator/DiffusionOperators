import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import math
import os 
import argparse

from timeit import default_timer

from utils import sigma_sequence, avg_spectrum, sample_trace
from random_fields_2d import PeriodicGaussianRF2d, GaussianRF_idct
# from models import FNO2d, UNO
from models import UNO

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob

device = torch.device('cuda:0')

L = 10              ##
sigma_1 = 1.0       ##
sigma_L = 0.01      ##
npad = 8            ##
sigma = sigma_sequence(sigma_1, sigma_L, L).to(device)
Ntest = 5           ##
d_co_domain = 32    ##

batch_size = 16     ##
epochs = 300        ##
record_int = 10     ##

s = 128 - 8
h = 2*math.pi/s

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def parse_args():
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument('--datadir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--record_interval", type=int, default=10)
    parser.add_argument("--L", type=int, default=10,
                        description="Number of noise scales (timesteps)")
    parser.add_argument("--sigma_1", type=float, default=1.0)
    parser.add_argument("--sigma_L", type=float, default=0.01)   
    parser.add_argument("--npad", type=int, default=8)
    parser.add_argument("--Ntest", type=int, default=5)
    args = parser.parse_args()
    return args

#Inport data

class VolcanoDataset(Dataset):

    def __init__(self, root, ntrain=4096):

        super().__init__()

        res = 128-8
        files = glob.glob("{}/**/*.int".format(root), recursive=True)[:ntrain]
        if len(files) == 0:
            raise Exception("Cannot find any *.int files here.")
        print("# files detected: {}".format(len(files)))

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

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idc):
        return self.x_train[idc]

    def __extra_repr__(self):
        return "shape={}, min={}, max={}".format(len(self.x_train), 
                                                 self.x_train.min(), 
                                                 self.x_train.max())

datadir = os.environ.get("DATA_DIR", None)
if datadir is None:
    raise ValueError("Environment variable DATA_DIR must be set")

train_dataset = VolcanoDataset(root=datadir)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

import pdb; pdb.set_trace()


# fno = FNO2d(s=s, width=64, modes=80, out_channels = 2, in_channels = 2)
fno = UNO(2+3, d_co_domain, s = s, pad=npad).to(device)
print("# of trainable parameters: {}".format(count_params(fno)))
fno = fno.to(device)
optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

# noise_sampler = PeriodicGaussianRF2d(s, s, alpha=1.5, tau=5, sigma=4.0, device=device)
# init_sampler = PeriodicGaussianRF2d(s, s, alpha=1.1, tau=0.1, sigma=1.0, device=device)
noise_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=1, sigma = 0.1, device=device)
# init_sampler = GaussianRF_idct(s, s, alpha=1.1, tau=0.1, sigma = 8, device=device)
init_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=1, sigma = 1, device=device)



#plt.imshow(init_sampler.sample(1).cpu().view(s,s))
#plt.colorbar()
#plt.savefig('test.png')


for ep in range(epochs):
    train_err = 0.0
    steps = 0
    t1 = default_timer()

    fno.train()
    pbar = tqdm(total=len(data_loader), desc="Epoch: {}".format(ep))
    for u in data_loader:
        optimizer.zero_grad()

        u = u.to(device)
        bsize = u.size(0)

        r = random.randint(0,L-1)
        noise = sigma[r]*noise_sampler.sample(bsize)
        
        # print('noise', noise.size())
        # print('u', u.size())
        # print('f(u)', fno(u, sigma[r].view(1,1)).size())
        # loss = ((h**2)/bsize)*((sigma[r]*torch.permute(fno(u + noise, sigma[r].view(1,1)), (0, 2, 3, 1)) + (1.0/sigma[r])*noise)**2).sum()

        # TODO: verify that this loss is correct
        loss = ((h**2)/bsize)*((sigma[r]*fno(u + noise, sigma[r].view(1,1)) + (1.0/sigma[r])*noise)**2).sum()

        

        loss.backward()
        pbar.update(1)
        pbar.set_postfix({
            'loss': loss.item()
        })

        optimizer.step()

        train_err += loss.item()
        steps += 1
    
    scheduler.step()

    train_err /= steps

    if (ep + 1) % record_int == 0:
        fno.eval()
        with torch.no_grad():
            u = init_sampler.sample(Ntest)
            u = sample_trace(fno, noise_sampler, sigma, u, epsilon=2e-5, T=400)
            u = u.view(Ntest,-1)
            u = u[~torch.any(u.isnan(),dim=1)]
            try:
                u = u.view(-1,s,s,2)
            except:
                continue
            

        #stats[k, (ep+1)//record_int -1, 0] = train_err
        #stats[k, (ep+1)//record_int -1, 1] = max_err
        #stats[k, (ep+1)//record_int -1, 2] = l2_err
        #stats[k, (ep+1)//record_int -1, 3] = Ntest - u.size(0)

        

        path  = 'ns_noise_400_point1_noise_15_1_UNO_init/' + str(s) + '/' + str(ep+1) + '/'
        path_Figure  = 'ns_noise_400_point1_noise_15_1_UNO_init/' + str(s) + 'Figure/' 
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_Figure):
            os.makedirs(path_Figure)

        torch.save(fno.state_dict(), path + 'model.pt')

        u = u.cpu()
        # for j in range(u.size(0)):
        #     plt.figure(j)
        #     plt.imshow(u[j,:,:].view(s,s))
        #     plt.colorbar()
        #     plt.savefig(path + str(j) + '.png')
        #     plt.close()
        fig, ax = plt.subplots(1, Ntest, figsize=(16,4))
        for j in range(u.size(0)):
            phase = torch.atan2(u[j,:,:,1], u[j,:,:,0]).cpu().detach().numpy()
            phase = (phase + np.pi) % (2 * np.pi) - np.pi
            bar = ax[j].imshow(phase,  cmap='RdYlBu', vmin = -np.pi, vmax=np.pi,extent=[0,1,0,1])
        cax = fig.add_axes([ax[Ntest-1].get_position().x1+0.01,ax[Ntest-1].get_position().y0,0.02,ax[Ntest-1].get_position().height])
        plt.colorbar(bar, cax=cax) # Similar to fig.colorbar(im, cax = cax)
        # print(path+'.pdf')
        plt.savefig(path_Figure + str(ep+1)+'.pdf')  
        
        print(ep+1, train_err, default_timer() - t1)
        
    else:
        print(ep+1, train_err, default_timer() - t1)

#scipy.io.savemat('gm_trace/stats.mat', {'stats': stats})
