import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision.utils import save_image
from torchvision import transforms
import random
import math
import os 
import argparse
import json
import pickle

from scipy.stats import wasserstein_distance as w_distance

from timeit import default_timer

from datasets import VolcanoDataset

from ema import EMAHelper

from utils import (sigma_sequence, 
                   avg_spectrum, 
                   sample_trace, 
                   sample_trace_jit,
                   DotDict, 
                   circular_skew,
                   circular_var,
                   plot_noise,
                   plot_noised_samples,
                   plot_samples)
from random_fields_2d import (PeriodicGaussianRF2d, 
                              GaussianRF_idct, 
                              IndependentGaussian)
# from models import FNO2d, UNO
from models import UNO

import numpy as np
#import scipy.io

from tqdm import tqdm

device = torch.device('cuda:0')

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
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Size of the validation set (as a float in [0,1])")
    parser.add_argument("--record_interval", type=int, default=100)
    parser.add_argument("--savedir", required=True, type=str)
    # Samplers and prior distribution
    parser.add_argument("--L", type=int, default=10,
                        help="Number of noise scales (timesteps)")
    parser.add_argument("--schedule", type=str, default="geometric",
                        choices=["geometric", "linear"])
    parser.add_argument("--white_noise", action='store_true',
                        help="If set, use independent Gaussian noise instead")
    parser.add_argument("--augment", action='store_true',
                        help="If set, use data augmentation")
    parser.add_argument("--sigma_1", type=float, default=1.0)
    parser.add_argument("--sigma_L", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=2e-5,
                        help="Learning rate in the SGLD sampling algorithm")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Larger tau gives rougher (noisier) noise")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Larger alpha gives smoother noise")
    parser.add_argument("--sigma_x0", type=float, default=1.0,
                        help="Variance of the prior distribution")
    parser.add_argument("--T", type=int, default=100,
                        help="The T parameter for annealed SGLD (how many iters per sigma)")
    # U-Net specific
    parser.add_argument("--mult_dims", type=eval, default="[1,2,4,4]")
    parser.add_argument("--fmult", type=float, default=0.25,
                        help="Multiplier for the number of Fourier modes per convolution." + \
                            "The number of modes will be set to int(dim/2 * fmult)")
    parser.add_argument("--d_co_domain", type=int, default=32,
                        help="Is this analogous to `dim` for a regular U-Net?")
    parser.add_argument("--npad", type=int, default=8)
    # Optimisation
    parser.add_argument("--lr", type=float, default=1e-3)
    # Evaluation
    parser.add_argument("--Ntest", type=int, default=1024,
                        help="Number of examples to generate for validation " + \
                            "(generating skew and variance metrics)")
    # Misc
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loader")
    parser.add_argument("--ema_rate", type=float, default=None)
    args = parser.parse_args()
    return args

#Inport data


@torch.no_grad()
def sample(fno, init_sampler, noise_sampler, sigma, n_examples, bs, T,
           epsilon=2e-5, 
           fns=None):

    buf = []
    if fns is not None:
        fn_outputs = {k:[] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    print(n_batches, n_examples, bs, "<<<")

    for _ in range(n_batches):
        u = init_sampler.sample(bs)
        res = u.size(1)
        u = sample_trace(fno, noise_sampler, sigma, u, epsilon=epsilon, T=T) # (bs, res, res, 2)
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
    if len(buf) != n_examples:
        print("WARNING: some NaNs were in the generated samples, there were only " + \
            "{} / {} valid samples generated".format(len(buf), n_examples))
    #assert len(buf) == n_examples
    return buf, fn_outputs

def score_matching_loss(fno, u, sigma, noise_sampler):
    """
    Notes:

    for x ~ p_{\sigma_i}(x|x0), and x0 ~ p_{data}(x):
      || \sigma_i*score_fn(x, \sigma_i) + (x - x0) / \sigma_i ||_2
    = || \sigma_i*score_fn(x0+noise, \sigma_i) + (x0 + noise - x0) / \sigma_i||_2
    = || \sigma_i*score_fn(x0+noise, \sigma_i) + (noise) / \sigma_i||_2

    NOTE: if we use the trick from "Improved techniques for SBGMs" paper then:
    score_fn(x0+noise, \sigma_i) = score_fn(x0+noise) / \sigma_i,
    which we can just implement inside the unet's forward() method:

    loss = || \sigma_i*(score_fn(x0+noise) / \sigma_i) + (noise) / \sigma_i||_2
    """

    bsize = u.size(0)
    # Sample a noise scale per element in the minibatch
    idcs = torch.randperm(sigma.size(0))[0:bsize].to(u.device)
    this_sigmas = sigma[idcs].view(-1, 1, 1, 1)
    # z ~ N(0,\sigma) <=> 0 + \sigma*eps, where eps ~ N(0,1) (noise_sampler).
    noise = this_sigmas * noise_sampler.sample(bsize)
    # term1 = score_fn(x0+noise)
    u_noised = u + noise
    term1 =  this_sigmas * fno(u_noised, idcs, this_sigmas)
    term2 =  noise / this_sigmas
    loss = ((term1+term2)**2).mean()
    return loss

def init_model(args):
    """Return the model and datasets"""

    # Create the savedir if necessary.
    savedir = args.savedir
    print("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Dataset generation.
    datadir = os.environ.get("DATA_DIR", None)
    if datadir is None:
        raise ValueError("Environment variable DATA_DIR must be set")
    if args.augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    else:
        transform = None
    full_dataset = VolcanoDataset(root=datadir, transform=transform)
    rnd_state = np.random.RandomState(seed=0)
    dataset_idcs = np.arange(0, len(full_dataset))
    rnd_state.shuffle(dataset_idcs)
    train_dataset = Subset(
        full_dataset, 
        dataset_idcs[0 : int(len(dataset_idcs)*(1-args.val_size)) ]
    )
    valid_dataset = Subset(
        full_dataset, 
        dataset_idcs[ int(len(dataset_idcs)*(1-args.val_size)) :: ]
    )
    print("Len of train / valid: {} / {}".format(len(train_dataset),
                                                 len(valid_dataset)))

    # Initialise the model
    s = 128 - 8
    fno = UNO(2+2, 
              args.d_co_domain, 
              s = s, 
              pad=args.npad, 
              fmult=args.fmult, 
              mult_dims=args.mult_dims).to(device)
    print(fno)
    print("# of trainable parameters: {}".format(count_params(fno)))
    fno = fno.to(device)

    ema_helper = None
    if args.ema_rate is not None:
        ema_helper = EMAHelper(mu=args.ema_rate)
        ema_helper.register(fno)

    # Load checkpoint here if it exists.
    start_epoch = 0
    if os.path.exists(os.path.join(savedir, "model.pt")):
        chkpt = torch.load(os.path.join(savedir, "model.pt"))
        fno.load_state_dict(chkpt['weights'])
        start_epoch = chkpt['stats']['epoch']
        if ema_helper is not None and hasattr(chkpt, 'ema_helper'):
            ema_helper.load_state_dict(chkpt['ema_helper'])
        print("Found checkpoint, resuming from epoch {}".format(start_epoch))

    # Initialise samplers.
    # TODO: make this and sigma part of the model, not outside of it.
    if args.white_noise:
        print("Using independent Gaussian noise...")
        noise_sampler = IndependentGaussian(s, s, sigma=1.0, device=device)
        init_sampler = IndependentGaussian(s, s, sigma=1.0, device=device)
    else:
        noise_sampler = GaussianRF_idct(s, s,
                                        alpha=args.alpha, 
                                        tau=args.tau,
                                        sigma = 1.0, 
                                        device=device)
        init_sampler = GaussianRF_idct(s, s, 
                                    alpha=args.alpha, 
                                    tau=args.tau, 
                                    sigma = args.sigma_x0,
                                    device=device)

    if args.sigma_1 < args.sigma_L:
        raise ValueError("sigma_1 < sigma_L, whereas sigmas should be monotonically " + \
            "decreasing. You probably need to switch these two arguments around.")

    if args.schedule == 'geometric':
        sigma = sigma_sequence(args.sigma_1, args.sigma_L, args.L).to(device)
    elif args.schedule == 'linear':
        sigma = torch.linspace(args.sigma_1, args.sigma_L, args.L).to(device)
    else:
        raise ValueError("Unknown schedule: {}".format(args.schedule))
    
    print("sigma[0]={:.4f}, sigma[-1]={:.4f} for {} timesteps".format(
        sigma[0],
        sigma[-1],
        args.L
    ))

    # TODO: this needs to be cleaned up badly
    return fno, \
        ema_helper, \
        start_epoch, \
        (train_dataset, valid_dataset), \
        (init_sampler, noise_sampler, sigma)


class ValidationMetric():
    def __init__(self):
        self.best = np.inf
    def update(self, x):
        """Return true if the metric is the best so far, else false"""
        if x < self.best:
            self.best = x
            return True
        return False
    def state_dict(self):
        return {'best': self.best}
    def load_state_dict(self, dd):
        self.best = dd['best']
 
def run(args):

    savedir = args.savedir

    # TODO: clean up
    fno, ema_helper, start_epoch, (train_dataset, valid_dataset), (init_sampler, noise_sampler, sigma) = \
        init_model(args)

    #with ema_helper:
    #    print("test")
          
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )

    init_samples = init_sampler.sample(5).cpu()
    for ext in ['png', 'pdf']:
        plot_noise(
            init_samples, 
            os.path.join(
                savedir, 
                "noise", 
                "init_samples_tau{}_alpha{}_sigma{}.{}".format(
                    args.tau, 
                    args.alpha,
                    args.sigma_1, 
                    ext
                )
            )
        )
    noise_samples = noise_sampler.sample(5).cpu()
    for ext in ['png', 'pdf']:
        plot_noise(
            noise_samples, 
            os.path.join(
                savedir, 
                "noise", 
                # implicit that sigma here == 1.0
                "noise_samples_tau{}_alpha{}.{}".format(
                    args.tau, 
                    args.alpha, 
                    ext
                )
            )
        )

    # Save config file
    with open(os.path.join(savedir, "config.json"), "w") as f:
        f.write(json.dumps(args))

    # Compute the circular variance and skew on the training set
    # and save this to the experiment folder.
    var_train = circular_var(train_dataset.dataset.x_train).numpy()
    skew_train = circular_skew(train_dataset.dataset.x_train).numpy()
    with open(os.path.join(savedir, "gt_stats.pkl"), "wb") as f:
        pickle.dump(dict(var=var_train, skew=skew_train), f)

    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr, foreach=True)
    print(optimizer)

    f_write = open(os.path.join(savedir, "results.json"), "a")
    metric_trackers = {
        'w_skew': ValidationMetric(), 
        'w_var': ValidationMetric(),
        "w_total": ValidationMetric()
    }
    
    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()

        fno.train()
        pbar = tqdm(total=len(train_loader), desc="Train {}/{}".format(ep+1, args.epochs))
        buf = dict()
        for iter_, u in enumerate(train_loader):
            optimizer.zero_grad()

            u = u.to(device)

            #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = score_matching_loss(fno, u, sigma, noise_sampler)

            loss.backward()
            optimizer.step()
            pbar.update(1)

            if ema_helper is not None:
                ema_helper.update(fno)
            
            metrics = dict(loss=loss.item())
            if iter_ % 10 == 0:
                pbar.set_postfix(metrics)

            # Update total statistics
            for k,v in metrics.items():
                if k not in buf:
                    buf[k] = []
                buf[k].append(v)

            #if iter_ == 10: # TODO add debug flag
            #    break

            if iter_ == 0 and ep == 0:
                with torch.no_grad():
                    idcs = torch.linspace(0, len(sigma)-1, 16).long().to(u.device)
                    this_sigmas = sigma[idcs]
                    noise = this_sigmas.view(-1, 1, 1, 1) * noise_sampler.sample(16)
                    #print("noise magnitudes: min={}, max={}".format(noise.min(),
                    #                                                noise.max()))
                    plot_noised_samples(
                        # Use the same example, and make a 4x4 grid of points
                        u[0:1].repeat(16, 1, 1, 1) + noise, 
                        outfile=os.path.join(savedir, "u_noised.png"), 
                        subtitles=[ "u + {:.3f}*z".format(x) for x in \
                            this_sigmas.cpu().numpy() ],
                        figsize=(8,8)
                    )
                    
                    plot_noised_samples(
                        # Use the same example, and make a 4x4 grid of points
                        init_sampler.sample(16),
                        outfile=os.path.join(savedir, "u_prior.png"), 
                        figsize=(8,8)
                    )

        pbar.close()

        fno.eval()
        buf_valid = dict(loss_valid=[])
        for iter_, u in enumerate(valid_loader):
            u = u.to(device)
            loss  = score_matching_loss(fno, u, sigma, noise_sampler)
            # Update total statistics
            buf_valid["loss_valid"].append(loss.item())
   
        #scheduler.step()

        recorded = False
        if (ep + 1) % args.record_interval == 0:
            recorded = True

            with ema_helper:
                # This context mgr automatically applies EMA
                u, fn_outs = sample(
                    fno, init_sampler, noise_sampler, sigma, 
                    bs=args.val_batch_size, n_examples=args.Ntest, T=args.T,
                    epsilon=args.epsilon,
                    fns={"skew": circular_skew, "var": circular_var}
                )
                skew_generated = fn_outs['skew']
                var_generated = fn_outs['var']

            # Dump this out to disk as well.
            w_skew = w_distance(skew_train, skew_generated)
            w_var = w_distance(var_train, var_generated)
            w_total = w_skew + w_var
            metric_vals = {"w_skew": w_skew,
                           "w_var": w_var,
                           "w_total": w_total}

            for ext in ['pdf', 'png']:
                plot_samples(
                    u[0:5], 
                    outfile=os.path.join(
                        savedir, 
                        "samples",
                        "{}.{}".format(ep+1, ext)
                    )
                )

            # Nikola's suggestion: print the mean sample for training
            # set and generated set.
            mean_samples = torch.cat(( 
                train_dataset.dataset.x_train.mean(dim=0, keepdim=True), 
                u.mean(dim=0, keepdim=True).detach().cpu()
            ), dim=0)
            plot_samples(
                mean_samples,  # of shape (2, res, res, 2)
                outfile=os.path.join(savedir, "samples", "mean_sample_{}.png".format(ep+1))
            )
                                                        
            # Keep track of each metric, and save the following:
            for metric_key, metric_val in metric_vals.items():
                if metric_trackers[metric_key].update(metric_val):
                    print("new best metric for {}: {:.3f}".format(metric_key, metric_val))
                    for ext in ['pdf', 'png']:
                        plot_samples(
                            u[0:5], 
                            outfile=os.path.join(
                                savedir, 
                                "samples",
                                "best_{}.{}".format(metric_key, ext)
                            ),                        
                            title=str(
                                {'epoch':ep+1, metric_key: "{:.3f}".format(metric_val)}
                            )
                        )
                    # TODO: refactor
                    torch.save(
                        dict(
                            weights=fno.state_dict(),
                            metrics={k:v.state_dict() for k,v in metric_trackers.items()},
                            ema_helper=ema_helper.state_dict()
                        ),
                        os.path.join(savedir, "model.{}.pt".format(metric_key))
                    )
                    with open(os.path.join(savedir, 
                                           "samples", "best_{}.pkl".format(metric_key)), "wb") as f:
                        pickle.dump(
                            dict(var=var_generated, skew=skew_generated), f
                        )
            
        else:
            pass

        buf = {k:np.mean(v) for k,v in buf.items()}
        buf.update({k:np.mean(v) for k,v in buf_valid.items()})
        buf["epoch"] = ep
        buf["lr"] = optimizer.state_dict()['param_groups'][0]['lr']
        buf["time"] = default_timer() - t1
        #buf["sched_lr"] = scheduler.get_lr()[0] # should be the same as buf.lr
        if recorded:
            buf["w_skew"] = w_skew
            buf["w_var"] = w_var
            buf["w_total"] = w_skew + w_var
        f_write.write(json.dumps(buf) + "\n")
        f_write.flush()
        print(json.dumps(buf))

        # Save checkpoints
        # TODO: refactor
        torch.save(
            dict(
                weights=fno.state_dict(),
                metrics={k:v.state_dict() for k,v in metric_trackers.items()},
                ema_helper=ema_helper.state_dict()
            ),
            os.path.join(savedir, "model.pt")
        )

if __name__ == '__main__':

    args = DotDict(vars(parse_args()))
    run(args)
