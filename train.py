import argparse
import json
import math
import os
import pickle
from timeit import default_timer
from dataclasses import asdict, dataclass, field
from typing import List, Union, Tuple, Dict
from omegaconf import OmegaConf as OC

import torch
from scipy.stats import wasserstein_distance as w_distance
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

from src.datasets import VolcanoDataset
from src.models import UNO_Diffusion
from src.util.ema import EMAHelper
from src.util.random_fields_2d import (GaussianRF_RBF, IndependentGaussian,
                                       PeriodicGaussianRF2d)
from src.util.setup_logger import get_logger
from src.util.utils import (DotDict, count_params, avg_spectrum, circular_skew, circular_var,
                            plot_matrix, plot_noise, plot_samples,
                            plot_samples_grid, sample_trace, sigma_sequence,
                            to_phase, ValidationMetric)

logger = get_logger(__name__)

import numpy as np
from tqdm import tqdm

# import scipy.io

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--override_cfg",
        action="store_true",
        help="If this is set, then if there already exists a config.json "
        + "in the directory defined by savedir, load that instead of args.cfg. "
        + "This should be set so that SLURM does the right thing if the job is restarted.",
    )
    args = parser.parse_args()
    return args


@dataclass
class Arguments:

    batch_size: int = 16            # training batch size
    val_batch_size: int = 512       # validation batch size
    epochs: int = 100               #
    val_size: float = 0.1           # size of validation set (e.g. 0.1 = 10%)
    record_interval: int = 100      # eval valid metrics every this many epochs
    white_noise: bool = False       # use white noise instead of RBF
    augment: bool = False           # perform light data augmentation

    schedule: str = None            # either 'geometric' or 'linear'

    resolution: Union[int, None] = None # dataset resolution

    epsilon: float = 2e-5           # step size to use during generation (SGLD)
    sigma_1: float = 1.0            # variance of largest noise distribution
    sigma_L: float = 0.01           # variance of smallest noise distribution
    T: int = 100                    # how many SGLD steps to do per noise schedule
    L: int = 10                     # how many noise schedules between sigma_1 and sigma_L
    lambda_fn: str = '1/s'          # what weighting function do we use?

    rbf_scale: float = 1.0          # scale parameter of the RBF kernel (determines smoothness)
    rbf_eps: float = 0.01           # stability term for cholesky decomposition of covariance C

    factorization: str = None       # factorization, a specific kwarg in FNOBlocks
    num_freqs_input: int = 0        # not used currently

    npad: int = 8

    scale_factor: float = 1.0       # if < 1, downsize dataset by this amount.
    d_co_domain: int = 32           # lift from 2 dims (x,y) to this many dimensions inside UNO
    mult_dims: List[int] = field(default_factory=lambda: [1, 2, 4, 4]) # ngf

    # multiplier to reduce the number of Fourier modes.
    # For instance, 1.0 means the maximal possible number will be used,
    # and 0.5 means only half will be used. Larger numbers correspond
    # to more parameters / memory.
    fmult: float = 0.25
    
    rank: float = 1.0               # factorisation coefficient
    groups: int = 0                 # number of groups for group norm

    lr: float = 1e-3                # learning rate for training
    Ntest: int = 1024               # number of samples to compute for validation metrics
    num_workers: int = 2            # number of cpu workers
    ema_rate: Union[float, None] = None # moving average coefficient for EMA
    dry_run: bool = False

def get_dataset(args: DotDict) -> Tuple[Dataset, Dataset]:
    """Return training and validation split of dataset

    While it is messy, we use just a monolithic args
      object here so that it is easy to initialise
      the train/val dataset from other files when
      we load the experiment cfg file in.

    """
    # Dataset generation.
    datadir = os.environ.get("DATA_DIR", None)
    if datadir is None:
        raise ValueError("Environment variable DATA_DIR must be set")
    transform = None
    if args.augment:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    logger.debug("resolution: {}".format(args.resolution))
    dataset = VolcanoDataset(
        root=datadir,
        resolution=args.resolution,
        transform=transform
    )
    rnd_state = np.random.RandomState(seed=0)
    dataset_idcs = np.arange(0, len(dataset))
    rnd_state.shuffle(dataset_idcs)
    train_dataset = Subset(
        dataset, dataset_idcs[0 : int(len(dataset_idcs) * (1 - args.val_size))]
    )
    valid_dataset = Subset(
        dataset, dataset_idcs[int(len(dataset_idcs) * (1 - args.val_size)) : :]
    )
    logger.info(
        "Len of train / valid: {} / {}".format(len(train_dataset), len(valid_dataset))
    )
    return train_dataset, valid_dataset


@torch.no_grad()
def sample(
    fno, noise_sampler, sigma, n_examples, bs, T, epsilon=2e-5, fns=None
):
    buf = []
    if fns is not None:
        fn_outputs = {k: [] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    for _ in range(n_batches):
        u = noise_sampler.sample(bs)
        u = sample_trace(
            fno, noise_sampler, sigma, u, epsilon=epsilon, T=T
        )  # (bs, res, res, 2)
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


def score_matching_loss(fno, u, sigma, noise_sampler, 
                        lambda_fn):

    bsize = u.size(0)
    # loss = \lambda(sigma) || F(u+noise) + noise ||^2
    # and noise = \sqrt(sigma_i) L \epsilon, \epsilon ~ N(0,I).
    noise = torch.sqrt(sigma) * noise_sampler.sample(bsize)
    term1 = fno(u+noise, sigma)
    term2 = noise

    res_sq = u.size(1) * u.size(2)
    terms_flattened = (term1 + term2).view(bsize, res_sq, -1)

    # shape (bs,)
    weights = lambda_fn(sigma).flatten()
    # shape (bs,)
    loss = (terms_flattened**2).mean(dim=(1,2))

    weighted_loss = (weights*loss).mean()

    return weighted_loss

def init_model(args, savedir, checkpoint="model.pt"):
    """Return the model and datasets"""

    # Create the savedir if necessary.
    logger.info("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    train_dataset, valid_dataset = get_dataset(args)

    # Initialise the model
    logger.debug("res: {}, dataset.x_train.shape = {}".format(
        args.resolution, train_dataset.dataset.x_train.shape
    ))
    fno = UNO_Diffusion(
        in_channels=2,
        out_channels=2,
        base_width=args.d_co_domain,
        spatial_dim=train_dataset.dataset.res,
        npad=args.npad,
        rank=args.rank,
        fmult=args.fmult,
        factorization=args.factorization,
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
        fno.load_state_dict(chkpt["weights"])
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
        logger.debug("Noise distribution: RBF noise")
        noise_sampler = GaussianRF_RBF(
            train_dataset.dataset.res, 
            train_dataset.dataset.res, 
            scale=args.rbf_scale, 
            eps=args.rbf_eps, 
            device=device
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
        val_metrics,
        (train_dataset, valid_dataset),
        (noise_sampler, sigma),
    )

@torch.no_grad()
def generate_loss_distribution(loader, fno, sigmas, noise_sampler, lambda_fn):
    """Evaluate the loss for each possible value of sigma.
      Useful to plot for diagnostics.

    Args:
        loader (_type_): _description_
        fno (_type_): _description_
        sigmas (_type_): _description_
        noise_sampler (_type_): _description_
    """
    buf = []
    # Construct noise distribution for plotting.
    _iter = iter(loader)
    for j in range(len(sigmas)):
        try:
            u = next(_iter).to(device)
            sampled_sigma = sigmas[j].expand(u.size(0), 1, 1, 1).to(device)
            loss = score_matching_loss(fno, u, sampled_sigma, noise_sampler, lambda_fn)
            buf.append(loss.item())
            print(buf[-1])
        except StopIteration:
            _iter = iter(loader)
        finally:
            pass
    return buf


def run(args: Arguments, savedir: str):

    # TODO: clean up
    (
        fno,
        ema_helper,
        start_epoch,
        val_metrics,
        (train_dataset, valid_dataset),
        (noise_sampler, sigma),
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

    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr, foreach=True)
    logger.debug("optimizer: {}".format(optimizer))

    f_write = open(os.path.join(savedir, "results.json"), "a")
    metric_trackers = {
        "w_skew": ValidationMetric(),
        "w_var": ValidationMetric(),
        "w_total": ValidationMetric(),
        "mean_image_l2": ValidationMetric(),
        "mean_image_phase_l2": ValidationMetric(),
    }
    if val_metrics is not None:
        for key in val_metrics:
            metric_trackers[key].load_state_dict(val_metrics[key])
            logger.debug("set tracker: {}.best = {}".format(key, val_metrics[key]))

    if args.lambda_fn == '1/s':
        lambda_fn = lambda x: 1. / x
    elif args.lambda_fn == '1/s^2':
        lambda_fn = lambda x: 1. / (x**2)
    elif args.lambda_fn == 's^2':
        lambda_fn = lambda x: (x**2)
    else:
        raise ValueError("lambda_fn must be either: '1/s', '1/s^2', or 's^2'")

    if args.dry_run:
        buf = generate_loss_distribution(
            train_loader, 
            fno, 
            sigma, 
            noise_sampler, 
            lambda_fn
        )
        return

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

            # Sample a noise scale per element in the minibatch
            idcs = torch.randperm(sigma.size(0))[0:u.size(0)].to(u.device)
            sampled_sigma = sigma[idcs].view(-1, 1, 1, 1)
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = score_matching_loss(
                fno, 
                u, 
                sampled_sigma, 
                noise_sampler,
                lambda_fn
            )

            """
            u, fn_outs = sample(
                fno,
                noise_sampler,
                sigma,
                bs=args.val_batch_size,
                n_examples=args.Ntest,
                T=args.T,
                epsilon=args.epsilon,
                fns={"skew": circular_skew, "var": circular_var},
            )
            """

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
                        noise_sampler.sample(16),
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

        metric_vals = {} # store validation metrics
        if (ep + 1) % args.record_interval == 0:

            with ema_helper:
                # This context mgr automatically applies EMA
                u, fn_outs = sample(
                    fno,
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
        buf.update(metric_vals)
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

    saved_cfg_file = os.path.join(args.savedir, "config.json")
    if os.path.exists(saved_cfg_file) and not args.override_cfg:
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
