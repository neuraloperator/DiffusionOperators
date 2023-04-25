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

from scipy.stats import wasserstein_distance as w_distance

from timeit import default_timer

from datasets import VolcanoDataset

from ema import EMAHelper

from utils import (
    sigma_sequence,
    avg_spectrum,
    sample_trace,
    sample_trace_jit,
    DotDict,
    circular_skew,
    circular_var,
    plot_noise,
    plot_samples_grid,
    plot_samples,
)
from random_fields_2d import PeriodicGaussianRF2d, GaussianRF_idct, IndependentGaussian

from models import Generator, Discriminator

import numpy as np

# import scipy.io

from tqdm import tqdm

from setup_logger import get_logger

logger = get_logger(__name__)

device = torch.device("cuda:0")


from dataclasses import dataclass, field
from typing import List, Union
from omegaconf import OmegaConf as OC

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    return args

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

@torch.no_grad()
def sample(
    g, noise_sampler, n_examples, bs, fns=None
):
    buf = []
    if fns is not None:
        fn_outputs = {k: [] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    print(n_batches, n_examples, bs, "<<<")

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
    term1 = this_sigmas * fno(u_noised, idcs, this_sigmas)
    term2 = noise / this_sigmas
    loss = ((term1 + term2) ** 2).mean()
    return loss

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

    # Dataset generation.
    datadir = os.environ.get("DATA_DIR", None)
    if datadir is None:
        raise ValueError("Environment variable DATA_DIR must be set")
    if args.augment:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        transform = None
    full_dataset = VolcanoDataset(root=datadir, transform=transform)
    rnd_state = np.random.RandomState(seed=0)
    dataset_idcs = np.arange(0, len(full_dataset))
    rnd_state.shuffle(dataset_idcs)
    train_dataset = Subset(
        full_dataset, dataset_idcs[0 : int(len(dataset_idcs) * (1 - args.val_size))]
    )
    valid_dataset = Subset(
        full_dataset, dataset_idcs[int(len(dataset_idcs) * (1 - args.val_size)) : :]
    )
    logger.info(
        "Len of train / valid: {} / {}".format(len(train_dataset), len(valid_dataset))
    )

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
        if ema_helper is not None and "ema_helper" in chkpt:
            logger.info("EMA enabled, loading EMA weights...")
            ema_helper.load_state_dict(chkpt["ema_helper"])
    else:
        if checkpoint != "model.pt":
            raise Exception("Cannot find checkpoint: {}".format(checkpoint))

    s = 128 - args.npad
    # Initialise samplers.
    # TODO: make this and sigma part of the model, not outside of it.
    if args.white_noise:
        logger.warning("Using independent Gaussian noise, NOT grf noise...")
        noise_sampler = IndependentGaussian(s, s, sigma=1.0, device=device)
    else:
        noise_sampler = GaussianRF_idct(
            s, s, alpha=args.alpha, tau=args.tau, sigma=1.0, device=device
        )

    # TODO: this needs to be cleaned up badly
    return (
        G,
        D,
        ema_helper,
        start_epoch,
        (train_dataset, valid_dataset),
        noise_sampler,
    )


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


def run(args, savedir):

    # TODO: clean up
    (
        G, D,
        ema_helper,
        start_epoch,
        (train_dataset, valid_dataset),
        noise_sampler,
    ) = init_model(args, savedir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
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
                "noise_samples_tau{}_alpha{}.{}".format(args.tau, args.alpha, ext),
            ),
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

    G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, foreach=True) #, weight_decay=1e-4)
    D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, foreach=True) #, weight_decay=1e-4)
    fn_loss = nn.BCEWithLogitsLoss()

    f_write = open(os.path.join(savedir, "results.json"), "a")
    metric_trackers = {
        "w_skew": ValidationMetric(),
        "w_var": ValidationMetric(),
        "w_total": ValidationMetric(),
    }

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
            mean_samples = torch.cat(
                (
                    train_dataset.dataset.x_train.mean(dim=0, keepdim=True),
                    u.mean(dim=0, keepdim=True).detach().cpu(),
                ),
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

    cfg_file = json.loads(open(args.cfg, "r").read())
    # structured() allows type checking
    conf = OC.structured(Arguments(**cfg_file))
    
    # Since type checking is already done, convert
    # it back ito a (dot-accessible) dictionary.
    run(DotDict(conf), args.savedir)
