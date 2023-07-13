import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision.utils import save_image
from torchvision import transforms
import random
import math
import os
import argparse
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Union
from omegaconf import OmegaConf as OC

from scipy.stats import wasserstein_distance as w_distance

from timeit import default_timer

from src.datasets import VolcanoDataset

from src.util.ema import EMAHelper
from src.util.utils import (
    sigma_sequence,
    avg_spectrum,
    sample_trace,
    sample_trace_jit,
    DotDict,
    circular_skew,
    circular_var,
    plot_matrix,
    plot_noise,
    plot_samples_grid,
    plot_samples,
    to_phase,
    ValidationMetric
)
from src.util.random_fields_2d import (GaussianRF_RBF, IndependentGaussian,
                                       PeriodicGaussianRF2d)
from src.models_gano import Generator, Discriminator

from train import get_dataset

import numpy as np

from tqdm import tqdm

from src.util.setup_logger import get_logger
logger = get_logger(__name__)

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    return args

"""
@dataclass
class Arguments:

    batch_size: int = 16
    val_batch_size: int = 512
    epochs: int = 100
    val_size: float = 0.1
    record_interval: int = 100
    white_noise: bool = False
    augment: bool = False
    n_critic: int = 10
    lambda_grad: float = 10.
    tau: float = 1.0
    alpha: float = 1.5
    sigma_x0: float = 1.0
    d_co_domain: int = 32
    npad: int = 8
    lr: float = 1e-3
    Ntest: int = 1024
    num_workers: int = 2
    ema_rate: Union[float, None] = None
"""

@dataclass
class Arguments:

    batch_size: int = 16            # training batch size
    val_batch_size: int = 512       # validation batch size
    epochs: int = 100               #
    val_size: float = 0.1           # size of validation set (e.g. 0.1 = 10%)
    record_interval: int = 100      # eval valid metrics every this many epochs
    white_noise: bool = False       # use white noise instead of RBF
    augment: bool = False           # perform light data augmentation

    resolution: Union[int, None] = None # dataset resolution
    npad: int = 8                   # how much input padding in UNO

    epsilon: float = 2e-5           # step size to use during generation (SGLD)
    sigma_1: float = 1.0            # variance of largest noise distribution
    sigma_L: float = 0.01           # variance of smallest noise distribution
    T: int = 100                    # how many SGLD steps to do per noise schedule
    L: int = 10                     # how many noise schedules between sigma_1 and sigma_L

    # GAN specific
    n_critic: int = 10              # number of D iters per G iter
    lambda_grad: float = 10.        # gradient penalty weighting

    rbf_scale: float = 1.0          # scale parameter of the RBF kernel (determines smoothness)
    rbf_eps: float = 0.01           # stability term for cholesky decomposition of covariance C

    # Currently not used for GANO
    #factorization: str = None       # factorization, a specific kwarg in FNOBlocks
    #num_freqs_input: int = 0        # not used currently

    scale_factor: float = 1.0       # if < 1, downsize dataset by this amount.
    d_co_domain: int = 32           # lift from 2 dims (x,y) to this many dimensions inside UNO

    # Currently not used in GANO
    #mult_dims: List[int] = field(default_factory=lambda: [1, 2, 4, 4]) # ngf

    # multiplier to reduce the number of Fourier modes.
    # For instance, 1.0 means the maximal possible number will be used,
    # and 0.5 means only half will be used. Larger numbers correspond
    # to more parameters / memory.
    fmult: float = 0.25
    
    rank: float = 1.0               # rank coefficient, a specific kwarg in FNOBlocks
    groups: int = 0                 # number of groups for group norm

    lr: float = 1e-3                # learning rate for training
    Ntest: int = 1024               # number of samples to compute for validation metrics
    num_workers: int = 2            # number of cpu workers
    ema_rate: Union[float, None] = None # moving average coefficient for EMA

@torch.no_grad()
def sample(
    g, noise_sampler, n_examples, bs, fns=None
):
    buf = []
    if fns is not None:
        fn_outputs = {k: [] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    for _ in range(n_batches):
        z = noise_sampler.sample(bs)
        u = g(z)
        if fns is not None:
            for fn_name, fn_apply in fns.items():
                fn_outputs[fn_name].append(fn_apply(u).cpu())
        buf.append(u.cpu())
    buf = torch.cat(buf, dim=0)[0:n_examples]
    # Flatten each list in fn outputs
    if fns is not None:
        fn_outputs = {
            k: torch.cat(v, dim=0)[0:n_examples] for k, v in fn_outputs.items()
        }
    #if len(buf) != n_examples:
    #    print(
    #        "WARNING: some NaNs were in the generated samples, there were only "
    #        + "{} / {} valid samples generated".format(len(buf), n_examples)
    #    )
    # assert len(buf) == n_examples
    return buf, fn_outputs

def calculate_gradient_penalty(model, real_images, fake_images, device, res):
    """Calculates the gradient penalty loss for GAN"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1.0/res) ** 2)
    return gradient_penalty


def init_model(args, savedir, checkpoint="model.pt"):
    """Return the model and datasets"""

    # Create the savedir if necessary.
    logger.info("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    train_dataset, valid_dataset = get_dataset(args)

    # Initialise the model
    D = Discriminator(2+2, args.d_co_domain, pad=args.npad).to(device)
    # TODO: why was it 2+1 to begin with???
    G = Generator(2+2, args.d_co_domain, pad=args.npad).to(device)
    nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    logger.info("Number discriminator parameters: {}".format(nn_params))
    nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    logger.info("Number generator parameters: {}".format(nn_params))

    ema_helper = None
    if args.ema_rate is not None:
        ema_helper = EMAHelper(mu=args.ema_rate)
        ema_helper.register(G)

    # Load checkpoint here if it exists.
    start_epoch = 0
    val_metrics = None
    if os.path.exists(os.path.join(savedir, checkpoint)):
        chkpt = torch.load(os.path.join(savedir, checkpoint))
        if "last_epoch" not in chkpt:
            start_epoch = 1
        else:
            start_epoch = chkpt["last_epoch"] + 1
        logger.info(
            "Found checkpoint {}, resuming from epoch {}".format(
                checkpoint, start_epoch
            )
        )
        logger.debug("keys in chkpt: {}".format(chkpt.keys()))
        G.load_state_dict(chkpt["g"])
        D.load_state_dict(chkpt["d"])
        logger.info("metrics found in chkpt: {}".format(chkpt["metrics"]))
        val_metrics = chkpt["metrics"]
        if ema_helper is not None and "ema_helper" in chkpt:
            logger.info("EMA enabled, loading EMA weights...")
            ema_helper.load_state_dict(chkpt["ema_helper"])
    else:
        if checkpoint != "model.pt":
            raise Exception("Cannot find checkpoint: {}".format(checkpoint))

    # Initialise samplers.
    # TODO: make this and sigma part of the model, not outside of it.
    if args.white_noise:
        logger.warning("Noise distribution: independent Gaussian noise")
        noise_sampler = IndependentGaussian(
            train_dataset.dataset.res, 
            train_dataset.dataset.res, 
            sigma=1.0, 
            device=device
        )
    else:
        logger.debug("Noise distribution: RBF noise, lambda={}".\
            format(args.rbf_scale))
        noise_sampler = GaussianRF_RBF(
            train_dataset.dataset.res, 
            train_dataset.dataset.res, 
            scale=args.rbf_scale, 
            eps=args.rbf_eps, 
            device=device
        )

    # TODO: this needs to be cleaned up badly
    return (
        G,
        D,
        ema_helper,
        start_epoch,
        val_metrics,
        (train_dataset, valid_dataset),
        noise_sampler,
    )

def run(args, savedir):

    # TODO: clean up
    (
        G, D,
        ema_helper,
        start_epoch,
        val_metrics,
        (train_dataset, valid_dataset),
        noise_sampler,
    ) = init_model(args, savedir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Currently not used.
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    noise_samples = noise_sampler.sample(5).cpu()
    for ext in ["png", "pdf"]:
        plot_noise(
            noise_samples,
            os.path.join(
                savedir,
                "noise",
                # implicit that sigma here == 1.0
                "noise_samples.{}".format(ext),
            ),
        )
        if hasattr(noise_sampler, 'C'):
            plot_matrix(
                noise_sampler.C[0:200, 0:200],
                os.path.join(
                    savedir,
                    "noise",
                    # implicit that sigma here == 1.0
                    "noise_sampler_C.{}".format(ext),
                ),
                title="noise_sampler.C[0:200,0:200]",
            )

    # Save config file
    with open(os.path.join(savedir, "config.json"), "w") as f:
        f.write(json.dumps(asdict(args)))

    # Compute the circular variance and skew on the training set
    # and save this to the experiment folder.
    var_train = circular_var(train_dataset.dataset.x_train).numpy()
    skew_train = circular_skew(train_dataset.dataset.x_train).numpy()
    with open(os.path.join(savedir, "gt_stats.pkl"), "wb") as f:
        pickle.dump(dict(var=var_train, skew=skew_train), f)

    G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, foreach=True) #, weight_decay=1e-4)
    D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, foreach=True) #, weight_decay=1e-4)

    f_write = open(os.path.join(savedir, "results.json"), "a")
    metric_trackers = {
        "w_skew": ValidationMetric(),
        "w_var": ValidationMetric(),
        "w_total": ValidationMetric(),
        "mean_image_l2": ValidationMetric(),
        "mean_image_phase_l2": ValidationMetric()
    }
    if val_metrics is not None:
        for key in val_metrics:
            metric_trackers[key].load_state_dict(val_metrics[key])
            logger.debug("set tracker: {}.best = {}".format(key, val_metrics[key]))


    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()

        G.train()
        D.train()
        
        pbar = tqdm(
            total=len(train_loader), desc="Train {}/{}".format(ep + 1, args.epochs)
        )
        buf = dict()
        for iter_, u in enumerate(train_loader):
            G_optim.zero_grad()
            D_optim.zero_grad()

            u = u.to(device)

            # Update D loss
            u_fake = G(noise_sampler.sample(u.size(0)))
            W_loss = -torch.mean(D(u)) + torch.mean(D(u_fake.detach()))
            grad_penalty = calculate_gradient_penalty(
                D, u.data, u_fake.data, device,
                res=u.size(-1)-args.npad
            )
            loss_D = W_loss + args.lambda_grad*grad_penalty
            loss_D.backward()
            D_optim.step()

            if iter_ == 0:
                logger.debug("u min and max: {}, {}".format(u.min(), u.max()))
                logger.debug("u_fake min and max: {}, {}".format(u_fake.min(), u_fake.max()))

            metrics = dict(loss_D=loss_D.item(),
                           loss_W=W_loss.item(),
                           loss_gp=grad_penalty.item())

            if (iter_ + 1) % args.n_critic == 0:
                G_optim.zero_grad()
                D_optim.zero_grad()

                u_fake_g = G(noise_sampler.sample(u.size(0)))

                loss_G = -torch.mean(D(u_fake_g))
                loss_G.backward()
                G_optim.step()

                metrics['loss_G'] = loss_G.item()

                #print("yes")

            pbar.update(1)

            if ema_helper is not None:
                ema_helper.update(G)

            if iter_ % 10 == 0:
                pbar.set_postfix(metrics)

            # Update total statistics
            for k, v in metrics.items():
                if k not in buf:
                    buf[k] = []
                buf[k].append(v)

            #if iter_ == 10: # TODO add debug flag
            #    break

        pbar.close()

        G.eval()
        D.eval()
        buf_valid = dict(loss_valid=[])
        """
        for iter_, u in enumerate(valid_loader):
            u = u.to(device)
            loss = score_matching_loss(fno, u, sigma, noise_sampler)
            # Update total statistics
            buf_valid["loss_valid"].append(loss.item())
        """

        # scheduler.step()

        recorded = False
        if (ep + 1) % args.record_interval == 0:
            recorded = True

            with ema_helper:
                # This context mgr automatically applies EMA
                u, fn_outs = sample(
                    G,
                    noise_sampler,
                    bs=args.val_batch_size,
                    n_examples=args.Ntest,
                    fns={"skew": circular_skew, "var": circular_var},
                )
                skew_generated = fn_outs["skew"]
                var_generated = fn_outs["var"]

            # Dump this out to disk as well.
            w_skew = w_distance(skew_train, skew_generated)
            w_var = w_distance(var_train, var_generated)
            w_total = w_skew + w_var
            metric_vals = {"w_skew": w_skew, "w_var": w_var, "w_total": w_total}

            for ext in ["pdf", "png"]:
                plot_samples(
                    u[0:5],
                    outfile=os.path.join(
                        savedir, "samples", "{}.{}".format(ep + 1, ext)
                    ),
                )

            # Nikola's suggestion: print the mean sample for training
            # set and generated set.
            this_train_mean = train_dataset.dataset.x_train.mean(dim=0, keepdim=True)
            this_gen_mean = u.mean(dim=0, keepdim=True).detach().cpu()

            mean_image_l2 = torch.mean((this_train_mean - this_gen_mean) ** 2)
            metric_vals["mean_image_l2"] = mean_image_l2.item()

            mean_image_phase_l2 = torch.mean(
                (to_phase(this_train_mean) - to_phase(this_gen_mean)) ** 2
            )
            metric_vals["mean_image_phase_l2"] = mean_image_phase_l2.item()

            mean_samples = torch.cat(
                (this_train_mean, this_gen_mean),
                dim=0,
            )
            plot_samples(
                mean_samples,  # of shape (2, res, res, 2)
                outfile=os.path.join(
                    savedir, "samples", "mean_sample_{}.png".format(ep + 1)
                ),
            )

            # Keep track of each metric, and save the following:
            for metric_key, metric_val in metric_vals.items():
                if metric_trackers[metric_key].update(metric_val):
                    print(
                        "new best metric for {}: {:.3f}".format(metric_key, metric_val)
                    )
                    for ext in ["pdf", "png"]:
                        plot_samples(
                            u[0:5],
                            outfile=os.path.join(
                                savedir, "samples", "best_{}.{}".format(metric_key, ext)
                            ),
                            title=str(
                                {
                                    "epoch": ep + 1,
                                    metric_key: "{:.3f}".format(metric_val),
                                }
                            ),
                        )
                    # TODO: refactor
                    torch.save(
                        dict(
                            g=G.state_dict(),
                            d=D.state_dict(),
                            metrics={
                                k: v.state_dict() for k, v in metric_trackers.items()
                            },
                            ema_helper=ema_helper.state_dict(),
                            last_epoch=ep,
                        ),
                        os.path.join(savedir, "model.{}.pt".format(metric_key)),
                    )
                    with open(
                        os.path.join(
                            savedir, "samples", "best_{}.pkl".format(metric_key)
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(dict(var=var_generated, skew=skew_generated), f)

        else:
            pass

        buf = {k: np.mean(v) for k, v in buf.items()}
        buf.update({k: np.mean(v) for k, v in buf_valid.items()})
        buf["epoch"] = ep
        buf["lr_g"] = G_optim.state_dict()["param_groups"][0]["lr"]
        buf["lr_d"] = D_optim.state_dict()["param_groups"][0]["lr"]        
        buf["time"] = default_timer() - t1
        # buf["sched_lr"] = scheduler.get_lr()[0] # should be the same as buf.lr
        if recorded:
            for k,v in metric_vals.items():
                buf[k] = v
            
        f_write.write(json.dumps(buf) + "\n")
        f_write.flush()
        print(json.dumps(buf))

        # Save checkpoints
        # TODO: refactor
        torch.save(
            dict(
                g=G.state_dict(),
                d=D.state_dict(),
                metrics={k: v.state_dict() for k, v in metric_trackers.items()},
                ema_helper=ema_helper.state_dict(),
                last_epoch=ep,
            ),
            os.path.join(savedir, "model.pt"),
        )

if __name__ == "__main__":
    args = parse_args()

    saved_cfg_file = os.path.join(args.savedir, "config.json")
    if os.path.exists(saved_cfg_file):
        cfg_file = json.loads(open(saved_cfg_file, "r").read())
        logger.debug("Found config in exp dir, loading instead...")
    else:
        cfg_file = json.loads(open(args.cfg, "r").read())

    # structured() allows type checking
    conf = OC.structured(Arguments(**cfg_file))

    # Since type checking is already done, convert
    # it back ito a (dot-accessible) dictionary.
    # (OC.to_object() returns back an Arguments object)
    run(OC.to_object(conf), args.savedir)
