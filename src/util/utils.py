import torch
import torch.fft as fft
from tqdm import tqdm
import numpy as np 

from typing import Tuple, List

import os
import matplotlib.pyplot as plt

class ValidationMetric:
    def __init__(self):
        self.best = np.inf

    def update(self, x):
        """Return true if the metric is the best so far, else false"""
        if x < self.best:
            self.best = x
            return True
        return False

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, dd):
        self.best = dd["best"]


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def to_phase(samples: torch.Tensor):
    assert samples.size(-1) == 2, "Last dim should be 2d"
    phase = torch.atan2(samples[...,1], samples[...,0])
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    return phase

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

def sigma_sequence(sigma_1, sigma_L, L):
    a = (sigma_L/sigma_1)**(1.0/(L-1))

    return torch.tensor([sigma_1*(a**l) for l in range(L)])

def avg_spectrum(u):
    s = u.size(1)

    return torch.abs(fft.rfft(u, norm='forward')[0:s//2]).mean(0)

def min_max_norm(u, eps=1e-6):
    # (x - min)
    # ---------  = z
    # (max-min)
    return (u - u.min()) / (u.max() - u.min() + eps)

def rescale(z, min_, max_):
    # x = z(max-min) + min
    return z*(max_-min_) + min_

def format_tuple(x, y):
    return "({:.3f}, {:.3f})".format(x, y)

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def sample_white(score, sigma, x0, epsilon=2e-5, T=200):
    L = sigma.size(0)

    for j in range(L):
        alpha = epsilon*((sigma[j]**2)/(sigma[-1])**2)
        for t in range(T):
            if j == L - 1 and t == T - 1:
                x0 = x0 + 0.5*alpha*score(x0, sigma[j].view(1,1))
            else:
                x0 = x0 + 0.5*alpha*score(x0, sigma[j].view(1,1)) + torch.sqrt(alpha)*torch.randn(x0.size(), device=x0.device)
        
    return x0

def sample_trace(score, noise_sampler, sigma, x0, epsilon=2e-5, T=100, verbose=True):
    L = sigma.size(0)
    if verbose:
        pbar = tqdm(total=L, desc="sample_trace()")
    for j in range(L):
        alpha = epsilon*((sigma[j]**2)/(sigma[-1])**2)
        curr_j = torch.LongTensor([j]*x0.size(0)).to(x0.device)
        for t in range(T):
            if j == L - 1 and t == T - 1:
                x0 = x0 + 0.5*alpha*score(x0, curr_j, sigma[j].view(1,1,1,1))
            else:
                x0 = x0 + 0.5*alpha*score(x0, curr_j, sigma[j].view(1,1,1,1)) + \
                    torch.sqrt(alpha)*noise_sampler.sample(x0.size(0))
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
        
    return x0

#@torch.jit.script
def sample_trace_jit(scorenet, noise_sampler, sigma, x0, epsilon=2e-5, T=100, verbose=True):
    L = len(sigma)
    T = int(T)
    for j in range(L):
        alpha = epsilon*((sigma[j]**2)/(sigma[-1])**2)
        curr_j = torch.LongTensor([j]*x0.size(0)).to(x0.device)
        for t in range(T):
            if j == L - 1 and t == T - 1:
                x0 = x0 + 0.5*alpha*scorenet(x0, curr_j, sigma[j].view(1,1,1,1))
            else:
                x0 = x0 + 0.5*alpha*scorenet(x0, curr_j, sigma[j].view(1,1,1,1)) + \
                    torch.sqrt(alpha)*noise_sampler.sample(x0.size(0))
        
    return x0

# Plotting functions

def plot_noise(samples: torch.Tensor, outfile: str, figsize=(16,4)):
    basedir = os.path.dirname(outfile)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    samples = samples.cpu().numpy()
    numb_fig = samples.shape[0]
    fig, ax = plt.subplots(1, numb_fig, figsize=figsize, squeeze=False)
    for i in range(numb_fig):
        bar = ax[0][i].imshow(samples[i,:,:,0], extent=[0,1,0,1])
    cax = fig.add_axes([ax[0][numb_fig-1].get_position().x1+0.01,
                        ax[0][numb_fig-1].get_position().y0,0.02,
                        ax[0][numb_fig-1].get_position().height])
    fig.colorbar(bar, cax=cax, orientation='vertical')
    fig.savefig(outfile, bbox_inches='tight')

def plot_matrix(matrix: torch.Tensor, outfile: str, title: str = None, figsize=(6,6)):
    assert len(matrix.shape) == 2, "matrix should be a 2d tensor"
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    this_plot = ax.matshow(matrix.cpu().numpy())
    cax = fig.add_axes([ax.get_position().x1+0.01,
                        ax.get_position().y0,0.02,
                        ax.get_position().height])
    fig.colorbar(this_plot, cax=cax, orientation='vertical')
    if title is not None:
        fig.suptitle(title)
    fig.savefig(outfile, bbox_inches='tight')

def plot_samples(samples: torch.Tensor, outfile: str, title: str = None, 
                 subtitles=None,
                 figsize=(16,4)):
    """LEGACY function"""
    basedir = os.path.dirname(outfile)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    ncol = samples.size(0)
    if subtitles is not None:
        assert len(subtitles) == ncol
    fig, ax = plt.subplots(1, ncol, figsize=figsize)
    for j in range(ncol):
        phase = to_phase(samples[j]).cpu().detach().numpy()
        bar = ax[j].imshow(phase,  
                           cmap='RdYlBu', 
                           vmin = -np.pi, 
                           vmax=np.pi,extent=[0,1,0,1])
        if subtitles is not None:
            ax[j].set_title(subtitles[j])
    cax = fig.add_axes(
        [ax[ncol-1].get_position().x1+0.01,
         ax[ncol-1].get_position().y0,0.02,
         ax[ncol-1].get_position().height]
    )
    if title is not None:
        fig.suptitle(title)
    fig.colorbar(bar, cax = cax, orientation='vertical')
    fig.savefig(outfile, bbox_inches='tight')

def plot_samples_grid(samples: torch.Tensor, 
                      outfile: str, 
                      nrow_ncol: Tuple[int] = None,
                      title: str = None,
                      subtitles: List[str] = None,
                      figsize: Tuple[float] = (16,4)):
    basedir = os.path.dirname(outfile)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    if nrow_ncol is None:
        nrow = ncol = int(np.sqrt(samples.size(0)))
    else:
        nrow = nrow_ncol[0]
        ncol = nrow_ncol[1]
    if subtitles is not None:
        assert len(subtitles) == ncol*nrow
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
    for i in range(nrow):
        for j in range(ncol):
            phase = to_phase(samples[i*nrow +j]).cpu().detach().numpy()
            ax[i][j].imshow(
                phase,  
                cmap='RdYlBu', 
                vmin = -np.pi, 
                vmax=np.pi,extent=[0,1,0,1]
            )
            if subtitles is not None:
                ax[i][j].set_title(subtitles[i*nrow + j])
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')
