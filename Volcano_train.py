import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import math
import os 
import argparse
import json
import pickle

from scipy.stats import wasserstein_distance as w_distance

from timeit import default_timer

from utils import sigma_sequence, avg_spectrum, sample_trace, DotDict, circular_skew, circular_var
from random_fields_2d import PeriodicGaussianRF2d, GaussianRF_idct
# from models import FNO2d, UNO
from models import UNO

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob

device = torch.device('cuda:0')

"""
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
"""

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def parse_args():
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument('--datadir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=512,
                        help="Batch size used for generating samples at inference time")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--record_interval", type=int, default=100)
    parser.add_argument("--L", type=int, default=10,
                        help="Number of noise scales (timesteps)")
    parser.add_argument("--sigma_1", type=float, default=1.0)
    parser.add_argument("--sigma_L", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Larger tau gives rougher (noisier) noise")
    parser.add_argument("--T", type=int, default=100,
                        help="The T parameter for annealed SGLD (how many iters per sigma)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--npad", type=int, default=8)
    parser.add_argument("--Ntest", type=int, default=1024)
    parser.add_argument("--d_co_domain", type=float, default=32)
    parser.add_argument("--savedir", required=True, type=str)
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

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idc):
        return self.x_train[idc]

    def __extra_repr__(self):
        return "shape={}, min={}, max={}".format(len(self.x_train), 
                                                 self.x_train.min(), 
                                                 self.x_train.max())

@torch.no_grad()
def sample(fno, init_sampler, noise_sampler, sigma, n_examples, bs, T, 
           fns=None):

    buf = []
    if fns is not None:
        fn_outputs = {k:[] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    print(n_batches, n_examples, bs, "<<<")

    for _ in range(n_batches):
        u = init_sampler.sample(bs)
        res = u.size(1)
        u = sample_trace(fno, noise_sampler, sigma, u, epsilon=2e-5, T=T) # (bs, res, res, 2)
        u = u.view(bs,-1) # (bs, res*res*2)
        u = u[~torch.any(u.isnan(),dim=1)]
        #try:
        u = u.view(-1,res,res,2) # (bs, res, res, 2)
        #except:
        #    continue
        if fns is not None:
            for fn_name, fn_apply in fns.items():
                fn_outputs[fn_name].append(fn_apply(u).cpu())
        buf.append(u.cpu())
    buf = torch.cat(buf, dim=0)[0:n_examples]
    # Flatten each list in fn outputs
    if fns is not None:
        fn_outputs = {k:torch.cat(v, dim=0)[0:n_examples] for k,v in fn_outputs.items()}
    
    assert len(buf) == n_examples
    return buf, fn_outputs

if __name__ == '__main__':

    args = DotDict(vars(parse_args()))

    savedir = args.savedir
    print("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    datadir = os.environ.get("DATA_DIR", None)
    if datadir is None:
        raise ValueError("Environment variable DATA_DIR must be set")
    train_dataset = VolcanoDataset(root=datadir)
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    s = 128 - 8
    h = 2*math.pi/s

    # fno = FNO2d(s=s, width=64, modes=80, out_channels = 2, in_channels = 2)
    fno = UNO(2+2, args.d_co_domain, s = s, pad=args.npad).to(device)
    print("# of trainable parameters: {}".format(count_params(fno)))
    fno = fno.to(device)
    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr)
    print(optimizer)

    # TODO: keep an eye on this
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

    # noise_sampler = PeriodicGaussianRF2d(s, s, alpha=1.5, tau=5, sigma=4.0, device=device)
    # init_sampler = PeriodicGaussianRF2d(s, s, alpha=1.1, tau=0.1, sigma=1.0, device=device)

    # z_t ~ N(0,I), as per annealed SGLD algorithm
    noise_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=args.tau, sigma = 1.0, device=device)
    # init x0, this can come from any distribution but lets also make it N(0,I).
    init_sampler = GaussianRF_idct(s, s, alpha=1.5, tau=args.tau, sigma = 1.0, device=device)

    if args.sigma_1 < args.sigma_L:
        raise ValueError("sigma_1 < sigma_L, whereas sigmas should be monotonically " + \
            "decreasing. You probably need to switch these two arguments around.")

    sigma = sigma_sequence(args.sigma_1, args.sigma_L, args.L).to(device)
    print("sigma[0]={:.4f}, sigma[-1]={:.4f} for {} timesteps".format(
        sigma[0],
        sigma[-1],
        args.L
    ))

    # Save config file
    with open(os.path.join(savedir, "config.json"), "w") as f:
        f.write(json.dumps(args))

    # Compute the circular variance and skew on the training set
    # and save this to the experiment folder.
    var_train = circular_var(train_dataset.x_train).numpy()
    skew_train = circular_skew(train_dataset.x_train).numpy()
    with open(os.path.join(savedir, "gt_stats.pkl"), "wb") as f:
        pickle.dump(dict(var=var_train, skew=skew_train), f)

    f_write = open(os.path.join(savedir, "results.json"), "a")
    for ep in range(args.epochs):
        t1 = default_timer()

        fno.train()
        pbar = tqdm(total=len(data_loader), desc="{}/{}".format(ep, args.epochs))
        buf = dict()
        for iter_, u in enumerate(data_loader):
            optimizer.zero_grad()

            u = u.to(device)
            bsize = u.size(0)

            # for x ~ p_{\sigma_i}(x|x0), and x0 ~ p_{data}(x):
            #   || \sigma_i*score_fn(x, \sigma_i) + (x - x0) / \sigma_i ||_2
            # = || \sigma_i*score_fn(x0+noise, \sigma_i) + (x0 + noise - x0) / \sigma_i||_2
            # = || \sigma_i*score_fn(x0+noise, \sigma_i) + (noise) / \sigma_i||_2
            # NOTE: if we use the trick from "Improved techniques for SBGMs" paper then:
            # score_fn(x0+noise, \sigma_i) = score_fn(x0+noise) / \sigma_i,
            # which we can just implement inside the unet's forward() method
            # loss = || \sigma_i*(score_fn(x0+noise) / \sigma_i) + (noise) / \sigma_i||_2

            # Sample a noise scale per element in the minibatch
            perm = torch.randperm(sigma.size(0))[0:bsize]
            this_sigmas = sigma[perm].view(-1, 1)
            # z ~ N(0,\sigma) <=> 0 + \sigma*eps, where eps ~ N(0,1) (noise_sampler).
            noise = this_sigmas.view(-1, 1, 1, 1) * noise_sampler.sample(bsize)
            # term1 = score_fn(x0+noise)
            term1 =  this_sigmas.view(-1, 1, 1, 1) * fno(u + noise, this_sigmas)
            term2 =  noise / this_sigmas.view(-1, 1, 1, 1)
            loss = ((term1+term2)**2).mean()

            loss.backward()
            optimizer.step()
            pbar.update(1)
            
            metrics = dict(loss=loss.item())
            if iter_ % 10 == 0:
                pbar.set_postfix(metrics)

            # Update total statistics
            for k,v in metrics.items():
                if k not in buf:
                    buf[k] = []
                buf[k].append(v)

            #if iter_ == 10:
            #    break

        pbar.close()        
        #scheduler.step()

        recorded = False
        if (ep + 1) % args.record_interval == 0:
            fno.eval()
            recorded = True

            u, fn_outs = sample(
                fno, init_sampler, noise_sampler, sigma, 
                bs=args.val_batch_size, n_examples=args.Ntest, T=args.T,
                fns={"skew": circular_skew, "var": circular_var}
            )
            skew_generated = fn_outs['skew']
            var_generated = fn_outs['var']
            # Dump this out to disk as well.
            with open(os.path.join(savedir, "stats.pkl"), "wb") as f:
                pickle.dump(dict(var=var_generated, skew=skew_generated), f)

            w_skew = w_distance(skew_train, skew_generated)
            w_var = w_distance(var_train, var_generated)
            w_total = w_skew + w_var

            print("w total:", w_total)

            #import pdb; pdb.set_trace()

            #stats[k, (ep+1)//record_int -1, 0] = train_err
            #stats[k, (ep+1)//record_int -1, 1] = max_err
            #stats[k, (ep+1)//record_int -1, 2] = l2_err
            #stats[k, (ep+1)//record_int -1, 3] = Ntest - u.size(0)
            
            path  = os.path.join(savedir,
                                 'ns_noise_400_point1_noise_15_1_UNO_init',
                                 str(s),
                                 str(ep+1))
            path_Figure = os.path.join(
                savedir,
                'ns_noise_400_point1_noise_15_1_UNO_init',
                str(s),
                'Figure' 
            )
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(path_Figure):
                os.makedirs(path_Figure)

            torch.save(fno.state_dict(), path + 'model.pt')

            #u = u.cpu()
            # for j in range(u.size(0)):
            #     plt.figure(j)
            #     plt.imshow(u[j,:,:].view(s,s))
            #     plt.colorbar()
            #     plt.savefig(path + str(j) + '.png')
            #     plt.close()

            #with torch.no_grad():
            #    var_generated = circular_var(u).numpy()
            #    skew_generated = circular_skew(u).numpy()

            Nplot = 5
            u_subset = u[0:Nplot]
            fig, ax = plt.subplots(1, Nplot, figsize=(16,4))
            for j in range(u_subset.size(0)):
                phase = torch.atan2(u_subset[j,:,:,1], u_subset[j,:,:,0]).cpu().detach().numpy()
                phase = (phase + np.pi) % (2 * np.pi) - np.pi
                bar = ax[j].imshow(phase,  cmap='RdYlBu', vmin = -np.pi, vmax=np.pi,extent=[0,1,0,1])
            cax = fig.add_axes([ax[Nplot-1].get_position().x1+0.01,
                                ax[Nplot-1].get_position().y0,0.02,
                                ax[Nplot-1].get_position().height])
            plt.colorbar(bar, cax=cax) # Similar to fig.colorbar(im, cax = cax)
            # print(path+'.pdf')
            plt.savefig(path_Figure + str(ep+1)+'.pdf', bbox_inches='tight')
            
            #print(ep+1, train_err, default_timer() - t1)
            
        else:
            #print(ep+1, train_err, default_timer() - t1)
            pass

        buf = {k:np.mean(v) for k,v in buf.items()}
        buf["epoch"] = ep
        buf["lr"] = optimizer.state_dict()['param_groups'][0]['lr']
        #buf["sched_lr"] = scheduler.get_lr()[0] # should be the same as buf.lr
        if recorded:
            buf["w_skew"] = w_skew
            buf["w_var"] = w_var
            buf["w_total"] = w_skew + w_var
        f_write.write(json.dumps(buf) + "\n")
        f_write.flush()
        print(json.dumps(buf))


    #scipy.io.savemat('gm_trace/stats.mat', {'stats': stats})
