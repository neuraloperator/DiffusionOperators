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
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

from src.datasets import VolcanoDataset, NavierStokesDataset
from src.models import UNO_Diffusion
from src.util.ema import EMAHelper
from src.util.random_fields_2d import (GaussianRF_RBF, IndependentGaussian,
                                       PeriodicGaussianRF2d)
from src.util.setup_logger import get_logger
from src.util.utils import (DotDict, count_params, avg_spectrum,
                            plot_matrix, plot_noise, plot_samples,
                            plot_samples_grid, sample_trace, sigma_sequence,
                            auto_suggest_sigma1,
                            auto_suggest_epsilon,
                            ValidationMetric)

logger = get_logger(__name__)

import numpy as np
from tqdm import tqdm

RUN_LOCAL = os.environ.get('RUN_LOCAL', False)
logger.debug("RUN_LOCAL={}".format(RUN_LOCAL))

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

DATASET_TO_CLASS = dict(
    volcano=VolcanoDataset,
    navier_stokes=NavierStokesDataset
)

@dataclass
class Arguments:

    # -- TRAINING --
    batch_size: int = 16                        # training batch size
    val_batch_size: int = 512                   # validation batch size
    epochs: int = 100                           # train for how many epochs
    record_interval: int = 100                  # eval valid metrics every this many epochs
    augment: bool = False                       # perform light data augmentation
    ema_rate: Union[float, None] = None         # moving average coefficient for EMA
    lr: float = 1e-3                            # learning rate for training
    num_workers: int = 2                        # number of cpu workers

    # -- DATASET --
    dataset: str = 'volcano'                    # name of dataset

    # -- EVALUATION --
    Ntest: int = 1024                           # number of samples to compute for validation metrics
        
    # -- DIFFUSION --
    white_noise: bool = False                   # use white noise instead of RBF
    schedule: str = None                        # either 'geometric' or 'linear'
    epsilon: float = 2e-5                       # step size to use during generation (SGLD)
    sigma_1: Union[float, None] = 1.0           # variance of largest noise distribution
    sigma_L: float = 0.01                       # variance of smallest noise distribution
    T: int = 100                                # how many corrector steps per predictor step
    L: int = 10                                 # how many noise schedules, len(sigmas)
    lambda_fn: str = 's^2'                      # what weighting function do we use?
    rbf_scale: float = 1.0                      # scale parameter of the RBF kernel (determines smoothness)
    rbf_eps: float = 0.01                       # stability term for cholesky decomp. of C

    # -- ARCHITECTURE --
    factorization: Union[str,None] = None       # factorization, type (see FNOBlocks)
    npad: int = 8                               # how much do we pad the original input?
    d_co_domain: int = 32                       # base width for UNO
    # NOT USED
    mult_dims: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    fmult: float = 0.25                         # what proportion of Fourier modes to retain
    rank: float = 1.0                           # factorisation coefficient
    groups: int = 0                             # NOT USED

    # do we use signal-noise ratio to determine step size during sampling?
    use_snr: bool = False                       

    log_sigmas: bool = True                     # log losses per value of sigma

def get_dataset(args: DotDict) -> Dataset:
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
    dataset_class = DATASET_TO_CLASS[args.dataset]
    
    dataset = dataset_class(
        root=datadir,
        transform=transform
    )
    return dataset

@torch.no_grad()
def sample(
    fno, noise_sampler, sigma, n_examples, bs, T, epsilon=2e-5
):
    buf = []
    n_batches = int(math.ceil(n_examples / bs))
    for _ in range(n_batches):
        # u_0 ~ N(0, sigma_max * C)
        u = noise_sampler.sample(bs) * sigma[0]
        u = sample_trace(
            fno, noise_sampler, sigma, u, epsilon=epsilon, T=T
        )  # (bs, res, res, 2)
        buf.append(u.cpu())
    buf = torch.cat(buf, dim=0)[0:n_examples]
    # assert len(buf) == n_examples
    return buf

def score_matching_loss_OLD(fno, u, sigma, noise_sampler, lambda_fn, verbose=False):
    bsize = u.size(0)
    # Sample a noise scale per element in the minibatch
    this_sigmas = sigma
    # noise = sqrt(sigma_i) * (L * epsilon)
    # loss = || noise + sigma_i * F(u+noise) ||^2
    Le = noise_sampler.sample(bsize)
    noise = this_sigmas * Le
    term1 = this_sigmas * fno(u+noise, this_sigmas)
    term2 = Le
    loss = ((term1+term2)**2).mean()
    return loss

def score_matching_loss(fno, u, sigma, noise_sampler, 
                        lambda_fn):
    bsize = u.size(0)
    # loss = \lambda(sigma) || F(u+noise; sigma) + Le/sqrt(sigma) ||^2
    # and noise = \sqrt(sigma_i) L \epsilon, \epsilon ~ N(0,I).
    eps_L = noise_sampler.sample(bsize)         # structured noise
    noise = torch.sqrt(sigma) * eps_L           # scaled by variance
    term1 = fno(u+noise, sigma)
    term2 = eps_L * (1. / torch.sqrt(sigma))    # see my eqn 12 overleaf

    res_sq = u.size(1) * u.size(2)

    # shape (bs, s^2, 2)
    terms_flattened = (term1 + term2).view(bsize, res_sq, -1)
    # shape (bs,)
    weights = lambda_fn(sigma).flatten()
    # shape (bs,)
    loss = (terms_flattened**2).mean(dim=(1,2))
    # compute mean over minibatch
    weighted_loss = (weights*loss)

    return weighted_loss

def init_model(args, savedir, checkpoint="model.pt"):
    """Return the model and datasets"""

    # Create the savedir if necessary.
    logger.info("savedir: {}".format(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    train_dataset = get_dataset(args)

    # Initialise the model
    logger.debug("dataset.x_train.shape = {}".format(
        train_dataset.X.shape
    ))
    fno = UNO_Diffusion(
        in_channels=train_dataset.n_in,
        out_channels=train_dataset.n_in,
        base_width=args.d_co_domain,
        spatial_dim=train_dataset.res,
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
            train_dataset.n_in,
            train_dataset.res, 
            train_dataset.res, 
            sigma=1.0,
            device=device
        )
    else:
        logger.debug("Noise distribution: RBF noise")
        noise_sampler = GaussianRF_RBF(
            train_dataset.n_in,
            train_dataset.res, 
            train_dataset.res, 
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
        sigma = sigma_sequence(args.sigma_1, args.sigma_L, args.L)
    elif args.schedule == "linear":
        sigma = torch.linspace(args.sigma_1, args.sigma_L, args.L)
    else:
        raise ValueError("Unknown schedule: {}".format(args.schedule))
    logger.info("Suggested epsilon: {}".format(auto_suggest_epsilon(sigma, args.T)))
    sigma = sigma.to(device)

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
        train_dataset,
        (noise_sampler, sigma),
    )

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
    #with open(os.path.join(savedir, "gt_stats.pkl"), "wb") as f:
    #    pickle.dump(dict(var=var_train, skew=skew_train), f)

    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr, foreach=True)
    logger.debug("optimizer: {}".format(optimizer))

    f_write = open(os.path.join(savedir, "results.json"), "a")
    metric_trackers = {}
    if val_metrics is not None:
        for key in val_metrics:
            metric_trackers[key].load_state_dict(val_metrics[key])
            logger.debug("set tracker: {}.best = {}".format(key, val_metrics[key]))

    lambda_fns = {
        '1/s': lambda x: 1. / x,
        '1/s^2': lambda x: 1. / (x**2),
        's^2': lambda x: (x**2),
        'sqrt(s)': lambda x: torch.sqrt(x),
        's': lambda x: x
    }
    if args.lambda_fn not in lambda_fns:
        raise Exception("lambda_fn must be one of: {}".format(
            list(lambda_fns.keys())
        ))
    else:
        lambda_fn = lambda_fns[args.lambda_fn]

    postproc = train_dataset.postprocess
    postproc_kwargs = train_dataset.postprocess_kwargs

    for ep in range(start_epoch, args.epochs):
        t1 = default_timer()
        eval_model = (ep + 1) % args.record_interval == 0

        pbar = tqdm(
            total=len(train_loader), desc="{}/{}".format(ep + 1, args.epochs)
        )
        buf = dict()                    # map strings to metrics
        fno.train()
        for iter_, u in enumerate(train_loader):
            optimizer.zero_grad()

            u = u.to(device)
            
            # Sample a noise scale per element in the minibatch
            idcs = torch.randperm(sigma.size(0))[0:u.size(0)].to(u.device)
            sampled_sigma = sigma[idcs].view(-1, 1, 1, 1)
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            losses = score_matching_loss_OLD(
                fno, 
                u, 
                sampled_sigma, 
                noise_sampler,
                lambda_fn,
                verbose=(iter_==0)
            )
            loss = losses.mean()


            """
            u  = sample(
                fno,
                noise_sampler,
                sigma,
                bs=args.val_batch_size,
                n_examples=args.Ntest,
                T=args.T,
                epsilon=args.epsilon,
                #fns={"skew": circular_skew, "var": circular_var},
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
                    logger.debug("used mem: {}, total: {}".format(used_mem, total_mem))

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

            if iter_ == 1: # TODO add debug flag
                break

            # TODO: refactor
            if iter_ == 0 and ep == 0:
                with torch.no_grad():
                    idcs = torch.linspace(0, len(sigma) - 1, 16).long().to(u.device)
                    this_sigmas = sigma[idcs]
                    noise = this_sigmas.view(-1, 1, 1, 1) * noise_sampler.sample(16)
                    noise = noise.to(u.device)
                    logger.info(os.path.join(savedir, "u_noised.png"))
                    plot_samples_grid(
                        # Use the same example, and make a 4x4 grid of points
                        postproc(u[0:1].repeat(16, 1, 1, 1) + noise),
                        outfile=os.path.join(savedir, "u_noised.png"),
                        subtitles=[
                            "u + {:.3f}*z".format(x) for x in this_sigmas.cpu().numpy()
                        ],
                        figsize=(8, 8),
                        imshow_kwargs=postproc_kwargs
                    )
                    logger.info(os.path.join(savedir, "u_prior.png"))
                    plot_samples_grid(
                        # Use the same example, and make a 4x4 grid of points
                        postproc(noise_sampler.sample(16)),
                        outfile=os.path.join(savedir, "u_prior.png"),
                        figsize=(8, 8),
                        imshow_kwargs=postproc_kwargs
                    )

        pbar.close()

        fno.eval()
        buf_valid = dict(loss_valid=[])
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
                u = sample(
                    fno,
                    noise_sampler,
                    sigma,
                    bs=args.val_batch_size,
                    n_examples=args.Ntest,
                    T=args.T,
                    epsilon=args.epsilon
                )
            metric_vals = train_dataset.evaluate(u)
            
            for ext in ["pdf", "png"]:
                plot_samples(
                    u[0:5],
                    outfile=os.path.join(
                        savedir, "samples", "{}.{}".format(ep + 1, ext)
                    ),
                )

            # Keep track of each metric, and save the following:
            for metric_key, metric_val in metric_vals.items():
                if metric_key not in metric_trackers:
                    logger.info("Detected metric {}, creating metric tracker...".\
                        format(metric_key))
                    metric_trackers[metric_key] = ValidationMetric()
                
                if metric_trackers[metric_key].update(metric_val):
                    print(
                        "new best metric for {}: {:.3f}".format(metric_key, metric_val)
                    )
                    for ext in ["pdf", "png"]:
                        plot_samples(
                            postproc(u[0:5]),
                            outfile=os.path.join(
                                savedir, "samples", "best_{}.{}".format(metric_key, ext)
                            ),
                            title=str(
                                {
                                    "epoch": ep + 1,
                                    metric_key: "{:.3f}".format(metric_val),
                                }
                            ),
                            imshow_kwargs=postproc_kwargs
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
                    """
                    with open(
                        os.path.join(
                            savedir, "samples", "best_{}.pkl".format(metric_key)
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(dict(var=var_generated, skew=skew_generated), f)
                    """
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
        if not RUN_LOCAL:
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
