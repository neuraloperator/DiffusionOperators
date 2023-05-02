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

# from models import FNO2d, UNO
from models import UNO

import numpy as np

# import scipy.io

from tqdm import tqdm

from setup_logger import get_logger

logger = get_logger(__name__)

device = torch.device("cuda:0")


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

from dataclasses import dataclass, field, asdict
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
    schedule: str = None

    white_noise: bool = False
    epsilon: float = 2e-5
    sigma_1: float = 1.0
    sigma_L: float = 0.01
    T: int = 100
    L: int = 10

    factorization: str = None
    num_freqs_input: int = 0
    
    d_co_domain: int = 32
    npad: int = 8
    mult_dims: List[int] = field(default_factory=lambda: [1,2,4,4])
    fmult: float = 0.25
    rank: float = 1.0
    groups: int = 0
    
    lr: float = 1e-3
    Ntest: int = 1024
    num_workers: int = 2
    ema_rate: Union[float, None] = None

@torch.no_grad()
def sample(
    fno, init_sampler, noise_sampler, sigma, n_examples, bs, T, epsilon=2e-5, fns=None
):
    buf = []
    if fns is not None:
        fn_outputs = {k: [] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    print(n_batches, n_examples, bs, "<<<")

    for _ in range(n_batches):
        u = init_sampler.sample(bs)
        res = u.size(1)
        u = sample_trace(
            fno, noise_sampler, sigma, u, epsilon=epsilon, T=T
        )  # (bs, res, res, 2)
        u = u.view(bs, -1)  # (bs, res*res*2)
        u = u[~torch.any(u.isnan(), dim=1)]
        # try:
        u = u.view(-1, res, res, 2)  # (bs, res, res, 2)
        # except:
        #    continue
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
    if len(buf) != n_examples:
        print(
            "WARNING: some NaNs were in the generated samples, there were only "
            + "{} / {} valid samples generated".format(len(buf), n_examples)
        )
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
    s = 128 - 8
    fno = UNO(
        2,
        args.d_co_domain,
        s=s,
        pad=args.npad,
        fmult=args.fmult,
        groups=args.groups,
        factorization=args.factorization,
        rank=args.rank,
        num_freqs_input=args.num_freqs_input,
        mult_dims=args.mult_dims,
    ).to(device)
    # (fno)
    logger.info("# of trainable parameters: {}".format(count_params(fno)))
    fno = fno.to(device)

    ema_helper = None
    if args.ema_rate is not None:
        ema_helper = EMAHelper(mu=args.ema_rate)
        ema_helper.register(fno)

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
        fno.load_state_dict(chkpt["weights"])
        logger.info("metrics found in chkpt: {}".format(chkpt["metrics"]))
        if ema_helper is not None and "ema_helper" in chkpt:
            logger.info("EMA enabled, loading EMA weights...")
            ema_helper.load_state_dict(chkpt["ema_helper"])
    else:
        if checkpoint != "model.pt":
            raise Exception("Cannot find checkpoint: {}".format(checkpoint))

    # Initialise samplers.
    # TODO: make this and sigma part of the model, not outside of it.
    if args.white_noise:
        logger.warning("Using independent Gaussian noise, NOT grf noise...")
        noise_sampler = IndependentGaussian(s, s, sigma=1.0, device=device)
        init_sampler = IndependentGaussian(s, s, sigma=1.0, device=device)
    else:
        noise_sampler = GaussianRF_idct(
            s, s, alpha=args.alpha, tau=args.tau, sigma=1.0, device=device
        )
        init_sampler = GaussianRF_idct(
            s, s, alpha=args.alpha, tau=args.tau, sigma=args.sigma_x0, device=device
        )

    if args.sigma_1 < args.sigma_L:
        raise ValueError(
            "sigma_1 < sigma_L, whereas sigmas should be monotonically "
            + "decreasing. You probably need to switch these two arguments around."
        )

    if args.schedule == "geometric":
        sigma = sigma_sequence(args.sigma_1, args.sigma_L, args.L).to(device)
    elif args.schedule == "linear":
        sigma = torch.linspace(args.sigma_1, args.sigma_L, args.L).to(device)
    else:
        raise ValueError("Unknown schedule: {}".format(args.schedule))

    logger.info(
        "sigma[0]={:.4f}, sigma[-1]={:.4f} for {} timesteps".format(
            sigma[0], sigma[-1], args.L
        )
    )

    # TODO: this needs to be cleaned up badly
    return (
        fno,
        ema_helper,
        start_epoch,
        (train_dataset, valid_dataset),
        (init_sampler, noise_sampler, sigma),
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


def run(args: Arguments, savedir: str):

    # TODO: clean up
    (
        fno,
        ema_helper,
        start_epoch,
        (train_dataset, valid_dataset),
        (init_sampler, noise_sampler, sigma),
    ) = init_model(args, savedir)

    # with ema_helper:
    #    print("test")

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

    init_samples = init_sampler.sample(5).cpu()
    for ext in ["png", "pdf"]:
        plot_noise(
            init_samples,
            os.path.join(
                savedir,
                "noise",
                "init_samples_tau{}_alpha{}_sigma{}.{}".format(
                    args.tau, args.alpha, args.sigma_1, ext
                ),
            ),
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
        f.write(json.dumps(asdict(args)))

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
        "w_skew": ValidationMetric(),
        "w_var": ValidationMetric(),
        "w_total": ValidationMetric(),
    }

    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()

        fno.train()
        pbar = tqdm(
            total=len(train_loader), desc="Train {}/{}".format(ep + 1, args.epochs)
        )
        buf = dict()
        for iter_, u in enumerate(train_loader):
            optimizer.zero_grad()

            u = u.to(device)

            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
            for k, v in metrics.items():
                if k not in buf:
                    buf[k] = []
                buf[k].append(v)

            # if iter_ == 10: # TODO add debug flag
            #    break

            if iter_ == 0 and ep == 0:
                with torch.no_grad():
                    idcs = torch.linspace(0, len(sigma) - 1, 16).long().to(u.device)
                    this_sigmas = sigma[idcs]
                    noise = this_sigmas.view(-1, 1, 1, 1) * noise_sampler.sample(16)
                    # print("noise magnitudes: min={}, max={}".format(noise.min(),
                    #                                                noise.max()))

                    logger.info(os.path.join(savedir, "u_noised.png"))
                    plot_samples_grid(
                        # Use the same example, and make a 4x4 grid of points
                        u[0:1].repeat(16, 1, 1, 1) + noise,
                        outfile=os.path.join(savedir, "u_noised.png"),
                        subtitles=[
                            "u + {:.3f}*z".format(x) for x in this_sigmas.cpu().numpy()
                        ],
                        figsize=(8, 8),
                    )

                    logger.info(os.path.join(savedir, "u_prior.png"))
                    plot_samples_grid(
                        # Use the same example, and make a 4x4 grid of points
                        init_sampler.sample(16),
                        outfile=os.path.join(savedir, "u_prior.png"),
                        figsize=(8, 8),
                    )

                    x_train = train_dataset.dataset.x_train
                    mean_samples = []
                    for _ in range(16):
                        perm = torch.randperm(len(x_train))[0:2048]
                        mean_samples.append(x_train[perm].mean(dim=0, keepdims=True))
                    mean_samples = torch.cat(mean_samples, dim=0)
                    logger.info(os.path.join(savedir, "mean_subsamples.png"))
                    plot_samples_grid(
                        mean_samples,
                        outfile=os.path.join(savedir, "mean_subsamples.png"),
                        figsize=(8, 8),
                        title="mean images over training set (size 2048 subsamples)",
                    )

        pbar.close()

        fno.eval()
        buf_valid = dict(loss_valid=[])
        for iter_, u in enumerate(valid_loader):
            u = u.to(device)
            loss = score_matching_loss(fno, u, sigma, noise_sampler)
            # Update total statistics
            buf_valid["loss_valid"].append(loss.item())

        # scheduler.step()

        recorded = False
        if (ep + 1) % args.record_interval == 0:
            recorded = True

            with ema_helper:
                # This context mgr automatically applies EMA
                u, fn_outs = sample(
                    fno,
                    init_sampler,
                    noise_sampler,
                    sigma,
                    bs=args.val_batch_size,
                    n_examples=args.Ntest,
                    T=args.T,
                    epsilon=args.epsilon,
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
                            weights=fno.state_dict(),
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
        buf["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
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
                weights=fno.state_dict(),
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
    # (OC.to_object() returns back an Arguments object)
    run(OC.to_object(conf), args.savedir)