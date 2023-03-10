import torch
import random
import math
import os 

from timeit import default_timer

from utils import sigma_sequence, avg_spectrum, sample_trace
from random_fields_2d import PeriodicGaussianRF2d, GaussianRF_idct
# from models import FNO2d, UNO
from models import UNO

import numpy as np
import scipy.io
# import matplotlib.pyplot as plt
import pylab as plt

device = torch.device('cuda:0')

L = 10
sigma_1 = 1.0
sigma_L = 0.01
sigma = sigma_sequence(sigma_1, sigma_L, L).to(device)
Ntest = 10

d_co_domain = 32 
npad = 8

batch_size = 16
epochs = 300
record_int = 10

s = 128 - 8
h = 2*math.pi/s



#Inport data
import glob
ntrain = 4096
res = 128-8
# files = glob.glob('/mount/data/InSAR_Volcano/**/*.int', recursive=True)[:ntrain]
files = glob.glob('/mount/data/InSAR_Volcano/**/*.int', recursive=True)[:ntrain]
x_train = torch.zeros(ntrain, res, res, 2).float()
for i, f in enumerate(files):
    dtype = np.float32
    nline = 128
    nsamp = 128

    with open(f, 'rb') as fn:
        load_arr = np.frombuffer(fn.read(), dtype=dtype)
        img = np.array(load_arr.reshape((nline, nsamp, -1)))

    phi = np.angle(img[:,:,0] + img[:,:,1]*1j)
    x_train[i,:,:,0] = torch.cos(torch.tensor(phi[:res, :res]))
    x_train[i,:,:,1] = torch.sin(torch.tensor(phi[:res, :res]))    




# data visualizaion
numb_fig = 4
ite = 200
fig, ax = plt.subplots(5,numb_fig, figsize=(14,16))
u = x_train[100:100+numb_fig*10,:,:,:]
u = u.view(numb_fig*10,-1)
u = u[~torch.any(u.isnan(),dim=1)]
u = u.view(-1,s,s,2)
for i in range(5):
    for j in range(numb_fig):
        phase = torch.atan2(u[i*numb_fig+j,:,:,1], u[i*numb_fig+j,:,:,0]).cpu().detach().numpy()
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        bar = ax[i,j].imshow(phase,  cmap='RdYlBu', vmin = -np.pi, vmax=np.pi,extent=[0,1,0,1])
plt.savefig('Figures/volcano_new/{}real.pdf'.format(ite))  
plt.close()


# fno = FNO2d(s=s, width=64, modes=80, out_channels = 2, in_channels = 2)
fno = UNO(2+3, d_co_domain, s = s, pad=npad).to(device)

# fno = fno.to(device)
# fno.load_state_dict(torch.load('ns_noise_200_4_UNO/120/300/model.pt'))
# fno.eval()

# noise_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=5, sigma = 32, device=device)
# init_sampler = GaussianRF_idct(s, s, alpha=1.1, tau=0.1, sigma = 8, device=device)

noise_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=1, sigma = 16, device=device)
init_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=1, sigma = 4, device=device)

# counter = 0
for ite in range(29):

    fno.load_state_dict(torch.load('ns_noise_200_16_same_noise_15_1_UNO/120/'+str((1+ite)*10)+'/model.pt'))
    # counter = (1+ite)*10+counter
    fno.eval()
    with torch.no_grad():
        u = init_sampler.sample(numb_fig*10)
        u = sample_trace(fno, noise_sampler, sigma, u, epsilon=2e-5, T=200)

        # u = u.view(numb_fig*10,-1)
        # u = u[~torch.any(u.isnan(),dim=1)]
        # u = u.view(-1,s,s,2)


        
    fig, ax = plt.subplots(5,numb_fig, figsize=(14,16))
    for i in range(5):
        for j in range(numb_fig):
            phase = torch.atan2(u[i*numb_fig+j,:,:,1], u[i*numb_fig+j,:,:,0]).cpu().detach().numpy()
            phase = (phase + np.pi) % (2 * np.pi) - np.pi
            bar = ax[i,j].imshow(phase,  cmap='RdYlBu', vmin = -np.pi, vmax=np.pi,extent=[0,1,0,1])
    plt.savefig('Figures/volcano_new/{}gen_200_16_same_noise_15_1_UNO.pdf'.format(ite))  
    plt.close()


#scipy.io.savemat('gm_trace/stats.mat', {'stats': stats})
