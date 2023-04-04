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

from utils import (sigma_sequence, 
                   avg_spectrum, 
                   sample_trace, 
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
import glob

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
    args = parser.parse_args()
    return args

#Inport data

class VolcanoDataset(Dataset):

    def __init__(self, root, ntrain=4096, transform=None):

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

    # Load checkpoint here if it exists.
    start_epoch = 0
    if os.path.exists(os.path.join(savedir, "model.pt")):
        chkpt = torch.load(os.path.join(savedir, "model.pt"))
        fno.load_state_dict(chkpt['weights'])
        start_epoch = chkpt['stats']['epoch']
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

    return fno, \
        start_epoch, \
        (train_dataset, valid_dataset), \
        (init_sampler, noise_sampler, sigma)
 
def run(args):

    savedir = args.savedir

    # TODO: clean up
    fno, start_epoch, (train_dataset, valid_dataset), (init_sampler, noise_sampler, sigma) = \
        init_model(args)
          
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

    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr)
    print(optimizer)

    f_write = open(os.path.join(savedir, "results.json"), "a")
    best_skew, best_var = np.inf, np.inf
    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()

        fno.train()
        pbar = tqdm(total=len(train_loader), desc="Train {}/{}".format(ep+1, args.epochs))
        buf = dict()
        for iter_, u in enumerate(train_loader):
            optimizer.zero_grad()

            u = u.to(device)

            loss = score_matching_loss(fno, u, sigma, noise_sampler)

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

            #if iter_ == 10: # TODO add debug flag
            #    break

            if iter_ == 0 and ep == 0:
                with torch.no_grad():
                    idcs = torch.linspace(0, len(sigma)-1, 16).long().to(u.device)
                    this_sigmas = sigma[idcs]
                    noise = this_sigmas.view(-1, 1, 1, 1) * noise_sampler.sample(16)
                    print("noise magnitudes: min={}, max={}".format(noise.min(),
                                                                    noise.max()))
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
                                                        
            # Keep track of best skew metric, and save
            # its samples.
            if w_skew < best_skew:
                best_skew = w_skew
                for ext in ['pdf', 'png']:
                    plot_samples(
                        u[0:5], 
                        outfile=os.path.join(
                            savedir, 
                            "samples",
                            "best_skew.{}".format(ext)
                        ),                        title=str(dict(epoch=ep+1, skew=best_skew))
                    )                         
                with open(os.path.join(savedir, "samples", "best_skew.pkl"), "wb") as f:
                    pickle.dump(
                        dict(var=var_generated, skew=skew_generated), f
                    )

            # Keep track of best var metric, and save
            # its samples.
            if w_var < best_var:
                best_var = w_var
                for ext in ['pdf', 'png']:
                    plot_samples(
                        u[0:5], 
                        outfile=os.path.join(
                            savedir, 
                            "samples",
                            "best_var.{}".format(ext)
                        ),
                        title=str(dict(epoch=ep+1, var=best_var))
                    )
                with open(os.path.join(savedir, "samples", "best_var.pkl"), "wb") as f:
                    pickle.dump(
                        dict(var=var_generated, skew=skew_generated), f
                    )
            
            #print(ep+1, train_err, default_timer() - t1)
            
        else:
            #print(ep+1, train_err, default_timer() - t1)
            pass

        buf = {k:np.mean(v) for k,v in buf.items()}
        buf.update({k:np.mean(v) for k,v in buf_valid.items()})
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

        # Save checkpoints
        torch.save(
            dict(weights=fno.state_dict(), stats=buf),
            os.path.join(savedir, "model.pt")
        )
        # Save early stopping checkpoints for skew and variance


    #scipy.io.savemat('gm_trace/stats.mat', {'stats': stats})

if __name__ == '__main__':

    args = DotDict(vars(parse_args()))
    run(args)
