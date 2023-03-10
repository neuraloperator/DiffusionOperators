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
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

L = 10
sigma_1 = 1.0
sigma_L = 0.01
npad = 8
sigma = sigma_sequence(sigma_1, sigma_L, L).to(device)
Ntest = 5
d_co_domain = 32

batch_size = 16
epochs = 300
record_int = 10

s = 128 - 8
h = 2*math.pi/s



#Inport data

import glob
ntrain = 4096
res = 128-8
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

# data = torch.load('/home/nikola/HDD/NavierStokes/2D/nsforcing_128_train.pt')['y']

data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)



# fno = FNO2d(s=s, width=64, modes=80, out_channels = 2, in_channels = 2)
fno = UNO(2+3, d_co_domain, s = s, pad=npad).to(device)
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
    for u in data_loader:
        optimizer.zero_grad()

        u = u[0].to(device)
        bsize = u.size(0)

        r = random.randint(0,L-1)
        noise = sigma[r]*noise_sampler.sample(bsize)
        print()
        # print('noise', noise.size())
        # print('u', u.size())
        # print('f(u)', fno(u, sigma[r].view(1,1)).size())
        # loss = ((h**2)/bsize)*((sigma[r]*torch.permute(fno(u + noise, sigma[r].view(1,1)), (0, 2, 3, 1)) + (1.0/sigma[r])*noise)**2).sum()
        loss = ((h**2)/bsize)*((sigma[r]*fno(u + noise, sigma[r].view(1,1)) + (1.0/sigma[r])*noise)**2).sum()

        loss.backward()

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
