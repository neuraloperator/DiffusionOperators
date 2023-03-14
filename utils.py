import torch
import torch.fft as fft
from tqdm import tqdm

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

def sigma_sequence(sigma_1, sigma_L, L):
    a = (sigma_L/sigma_1)**(1.0/(L-1))

    return torch.tensor([sigma_1*(a**l) for l in range(L)])

def avg_spectrum(u):
    s = u.size(1)

    return torch.abs(fft.rfft(u, norm='forward')[0:s//2]).mean(0)


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
        for t in range(T):
            if j == L - 1 and t == T - 1:
                x0 = x0 + 0.5*alpha*score(x0, sigma[j].view(1,1))
            else:
                x0 = x0 + 0.5*alpha*score(x0, sigma[j].view(1,1)) + torch.sqrt(alpha)*noise_sampler.sample(x0.size(0))
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
        
    return x0

    