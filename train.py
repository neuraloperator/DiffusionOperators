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
                            auto_suggest_sigma1,
                            auto_suggest_epsilon,
                            to_phase, ValidationMetric)

logger = get_logger(__name__)

from functools import partial

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

    batch_size: int = 16                        # training batch size
    val_batch_size: int = 512                   # validation batch size
    epochs: int = 100                           # train for how many epochs
    val_size: float = 0.1                       # NOT USED
    record_interval: int = 100                  # eval valid metrics every this many epochs
    white_noise: bool = False                   # use white noise instead of RBF
    augment: bool = False                       # perform light data augmentation

    resolution: Union[int, None] = None         # dataset resolution

    epsilon: float = 2e-5                       # step size to use during generation (SGLD)
    sigma_1: Union[float, None] = 1.0           # variance of largest noise distribution
    sigma_L: float = 0.01                       # variance of smallest noise distribution
    T: int = 100                                # how many corrector steps per predictor step
    L: int = 10                                 # how many noise schedules, len(sigmas)
    lambda_fn: str = 's^2'                      # what weighting function do we use?

    rbf_scale: float = 1.0                      # scale parameter of the RBF kernel (determines smoothness)
    rbf_eps: float = 0.01                       # stability term for cholesky decomp. of C

    factorization: Union[str,None] = None       # factorization, type (see FNOBlocks)
    npad: int = 8                               # how much do we pad the original input?
    d_co_domain: int = 32                       # base width for UNO
    # NOT USED
    mult_dims: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    fmult: float = 0.25                         # what proportion of Fourier modes to retain
    rank: float = 1.0                           # factorisation coefficient
    groups: int = 0                             # NOT USED
    norm: Union[str,None] = None
    num_freqs: int = 32                         # number of frequency features for sigma

    # do we use signal-noise ratio to determine step size during sampling?
    use_snr: bool = False                       

    lr: float = 1e-3                            # learning rate for training
    Ntest: int = 1024                           # number of samples to compute for validation metrics
    num_workers: int = 2                        # number of cpu workers
    log_sigmas: bool = True                     # log losses per value of sigma
    ema_rate: Union[float, None] = None         # moving average coefficient for EMA

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
    return dataset


@torch.no_grad()
def sample(
    G, noise_sampler, sigma, n_examples, bs, T, epsilon=2e-5, fns=None
):
    buf = []
    if fns is not None:
        fn_outputs = {k: [] for k in fns.keys()}

    n_batches = int(math.ceil(n_examples / bs))

    for _ in range(n_batches):
        u = noise_sampler.sample(bs)
        u = sample_trace(
            G, noise_sampler, sigma, u, epsilon=epsilon, T=T
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

def score_matching_loss_edm(F, u, sigma, noise_sampler):

    sigma = sigma.view(-1, 1, 1, 1).to(u.device)
    
    bsize = u.size(0)
    eps_L = noise_sampler.sample(bsize)         # structured noise
    noise = sigma * eps_L           # scaled by variance

    # TODO: does cin(sigma)*(u+noise) have unit variance?
    # see Eqn. (114) of EDM.
    term1 = F( cin(sigma)*(u + noise), cnoise(sigma) )

    # TODO: does the effective training target also have
    # unit variance?
    term2 = (1. / cout(sigma)) * (u - (cskip(sigma)*(u + noise)))

    loss = ((term1-term2)**2).mean(dim=(1,2,3))
    
    return loss

def cskip(sigma, sigma_data=0.7):
    return sigma_data**2 / (sigma**2 + sigma_data**2)

def cout(sigma, sigma_data=0.7):
    return (sigma*sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)

def cin(sigma, sigma_data=0.7):
    return 1. / torch.sqrt(sigma**2 + sigma_data**2)

def cnoise(sigma):
    return 0.25*torch.log(sigma)

def init_model(args, savedir, checkpoint="model.pt"):
    """Return the model and datasets"""

    # Create the savedir if necessary.
    logger.info("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    train_dataset = get_dataset(args)

    # Initialise the model
    logger.debug("res: {}, dataset.x_train.shape = {}".format(
        args.resolution, train_dataset.x_train.shape
    ))
    fno = UNO_Diffusion(
        in_channels=2,
        out_channels=2,
        base_width=args.d_co_domain,
        norm=args.norm,
        spatial_dim=train_dataset.res,
        npad=args.npad,
        rank=args.rank,
        fmult=args.fmult,
        num_freqs=args.num_freqs,
        factorization=args.factorization,
    ).to(device)
    # (fno)
    logger.info("# of trainable parameters: {}".format(count_params(fno)))
    fno = fno.to(device)

    ema_helper = None
    if args.ema_rate is not None:
        ema_helper = EMAHelper(mu=args.ema_rate)
        ema_helper.register(fno)

    if args.sigma_1 is None:
        args.sigma_1 = float(int(auto_suggest_sigma1(train_dataset.X)))
        logger.debug("sigma_1 is None, auto-suggested value is: {}".format(args.sigma_1))

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
            train_dataset.res, 
            train_dataset.res, 
            sigma=1.0, 
            device=device
        )
    else:
        logger.debug("Noise distribution: RBF noise")
        noise_sampler = GaussianRF_RBF(
            train_dataset.res, 
            train_dataset.res, 
            scale=args.rbf_scale, 
            eps=args.rbf_eps, 
            device=device
        )

    sigma = sigma_sequence(args.sigma_1, args.sigma_L, args.L).to(device)

    if args.sigma_1 < args.sigma_L:
        raise ValueError(
            "sigma_1 < sigma_L, whereas sigmas should be monotonically "
            + "decreasing. You probably need to switch these two arguments around."
        )

    # TODO: this needs to be cleaned up badly
    return (
        fno,
        ema_helper,
        start_epoch,
        val_metrics,
        train_dataset,
        (noise_sampler, sigma)
    )

def sample_sigma(bs, P_mean=-1.2, P_std=1.2):
    """sample ln sigma ~ N(P_mean, P_std)"""
    ln_sigma = torch.zeros((bs, 1)).normal_(P_mean, P_std)
    return torch.exp(ln_sigma)

def G(u: torch.FloatTensor, sigma: torch.FloatTensor, F=None):
    """we assume sigma here is 0-d and take care of it"""
    sigma = sigma.view(-1, 1, 1, 1).repeat(u.size(0), 1, 1, 1)
    return cskip(sigma)*u + cout(sigma)*F( cin(sigma)*u, cnoise(sigma) )

def run(args: Arguments, savedir: str):

    # TODO: clean up
    (
        fno,
        ema_helper,
        start_epoch,
        val_metrics,
        train_dataset,
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
    var_train = circular_var(train_dataset.x_train).numpy()
    skew_train = circular_skew(train_dataset.x_train).numpy()
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

    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()
        eval_model = (ep + 1) % args.record_interval == 0

        pbar = tqdm(
            total=len(train_loader), desc="Train {}/{}".format(ep + 1, args.epochs)
        )
        buf = dict()                    # map strings to metrics
        sigma_to_loss = dict()          # map sigma value to loss
        fno.train()
        for iter_, u in enumerate(train_loader):
            optimizer.zero_grad()
            u = u.to(device)
            sampled_sigma = sample_sigma(u.size(0))
            losses = score_matching_loss_edm(
                fno, 
                u, 
                sampled_sigma, 
                noise_sampler,
            )
            loss = losses.mean()

            if eval_model:
                # If this is an evaluation epoch, log sigma -> loss for the entire
                # data loader.
                with torch.no_grad():
                    for ss, vv in zip(sampled_sigma.cpu().flatten(), losses.cpu()):
                        ss = ss.item()
                        vv = vv.item()
                        if ss not in sigma_to_loss:
                            sigma_to_loss[ss] = []
                        sigma_to_loss[ss].append(vv)

            """
            u, fn_outs = sample(
                partial(G, F=fno),
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

            if iter_ == 0:
                with torch.cuda.device(device):
                    mem_info = torch.cuda.mem_get_info()
                    free_mem = mem_info[0] / 1024. / 1024.
                    total_mem = mem_info[1] / 1024. / 1024.
                    used_mem = total_mem - free_mem

            if ema_helper is not None:
                ema_helper.update()

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

            # TODO: refactor

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

                    x_train = train_dataset.x_train
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
        #buf_valid = dict(loss_valid=[])
        buf_valid = {}
        """
        for iter_, u in enumerate(valid_loader):
            u = u.to(device)
            # Sample a noise scale per element in the minibatch
            idcs = torch.randperm(sigma.size(0))[0:u.size(0)].to(u.device)
            sampled_sigma = sigma[idcs].view(-1, 1, 1, 1)
            loss = score_matching_loss(fno, u, sampled_sigma, noise_sampler, lambda_fn)
            # Update total statistics
            buf_valid["loss_valid"].append(loss.item())
        """
        # scheduler.step()

        metric_vals = {} # store validation metrics
        if eval_model:

            with ema_helper:
                # This context mgr automatically applies EMA
                u, fn_outs = sample(
                    partial(G, F=fno),
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

            # Dump sigma_to_losses out to disk.
            with open(os.path.join(savedir, "samples", "{}_s2l.pkl".format(ep+1)), "wb") as f:
                pickle.dump(sigma_to_loss, f)

            # Nikola's suggestion: print the mean sample for training
            # set and generated set.
            this_train_mean = train_dataset.x_train.mean(dim=0, keepdim=True)
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
        buf["gpu_used_mem"] = used_mem
        buf["gpu_total_mem"] = total_mem
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
