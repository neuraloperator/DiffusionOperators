import sys
import torch
import os
import json
import argparse
import numpy as np
import pickle
from src.util.utils import (
    DotDict,
    circular_skew,
    circular_var,
    plot_samples,
    plot_samples_grid,
    min_max_norm,
    format_tuple,
    rescale,
)

from train import init_model, sample, get_dataset

from src.util.setup_logger import get_logger
logger = get_logger(__name__)

from src.util.random_fields_2d import GaussianRF_RBF

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Full path of the experiment: <savedir>/<group>/<id>",
    )
    parser.add_argument("--savedir", type=str, required=True, help="dump stats here")
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "plot", "superres"],
        default="generate",
        help="Which mode to use?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=512,
        help="Batch size used for generating samples at inference time",
    )
    parser.add_argument(
        "--Ntest",
        type=int,
        default=1024,
        help="Number of examples to generate for validation "
        + "(generating skew and variance metrics)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = DotDict(vars(parse_args()))

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    expdir = args.exp_name
    cfg = DotDict(json.loads(open(os.path.join(expdir, "config.json"), "r").read()))
    # print(cfg)

    print(json.dumps(cfg, indent=4))

    (
        fno,
        ema_helper,
        start_epoch,
        val_metrics,
        (train_dataset, valid_dataset),
        (noise_sampler, sigma),
    ) = init_model(cfg, expdir, args.checkpoint)

    if args.mode == "generate":

        # Generate samples and dump them out to <savedir>/samples.pkl.

        with ema_helper:
            u, fn_outs = sample(
                fno,
                noise_sampler,
                sigma,
                bs=args.val_batch_size,
                n_examples=args.Ntest,
                T=cfg.T,
                epsilon=cfg.epsilon,
                fns={"skew": circular_skew, "var": circular_var},
            )
            skew_generated = fn_outs["skew"]
            var_generated = fn_outs["var"]

        # samples.pkl contains everything
        torch.save(
            (u.cpu(), skew_generated, var_generated),
            os.path.join(args.savedir, "samples.{}.pkl".format(args.checkpoint)),
        )

        # stats.pkl just contains skew and variance
        with open(
            os.path.join(args.savedir, "stats.{}.pkl".format(args.checkpoint)), "wb"
        ) as f:
            pickle.dump(dict(var=var_generated, skew=skew_generated), f)

    elif args.mode == 'superres':

        logger.info("Initialise 2x dataset and save stats...")
        
        # We need to initialise a version of the train_dataset that is
        # twice as large, since this will be our ground truth dataset.
        cfg_2x = DotDict(cfg.copy())
        cfg_2x['resolution'] *= 2       # e.g. 64px -> 128px
        cfg_2x['npad'] *= 2             # e.g. 4 -> 8
        train_dataset_2x, _ = get_dataset(cfg_2x)
        # Compute the circular variance and skew on the training set
        # and save this to the experiment folder.
        var_train = circular_var(train_dataset_2x.dataset.x_train).numpy()
        skew_train = circular_skew(train_dataset_2x.dataset.x_train).numpy()
        with open(os.path.join(args.savedir, "gt_stats_2x.pkl"), "wb") as f:
            pickle.dump(dict(var=var_train, skew=skew_train), f)

        logger.info("Generating...")

        res = train_dataset.dataset.res

        noise_sampler_2x = GaussianRF_RBF(
            res*2, res*2, scale=cfg.rbf_scale, eps=cfg.rbf_eps, device=device
        )

        # HACK: we need to double the padding attribute
        # in model
        fno.padding *= 2 # e.g. from 4px for 60px -> 8px for 120px
        with ema_helper:
            u, fn_outs = sample(
                fno,
                noise_sampler_2x,
                sigma,
                bs=args.val_batch_size,
                n_examples=args.Ntest,
                T=cfg.T,
                epsilon=cfg.epsilon,
                fns={"skew": circular_skew, "var": circular_var},
            )
            skew_generated = fn_outs["skew"]
            var_generated = fn_outs["var"]

        # samples.pkl contains everything
        torch.save(
            (u.cpu(), skew_generated, var_generated),
            os.path.join(args.savedir, "samples_2x.{}.pkl".format(args.checkpoint)),
        )
        # stats.pkl just contains skew and variance
        with open(
            os.path.join(args.savedir, "stats_2x.{}.pkl".format(args.checkpoint)), "wb"
        ) as f:
            pickle.dump(dict(var=var_generated, skew=skew_generated), f)

    elif args.mode == "plot":

        for postfix in ["", "_2x"]:

            pkl_filename = os.path.join(
                args.savedir, 
                "samples{}.{}.pkl".format(postfix, args.checkpoint)
            )
            logger.info(pkl_filename)
            if not os.path.exists(pkl_filename):
                logger.debug("Cannot find {}, skipping...".format(pkl_filename))
                continue

            samples, skew_generated, var_generated = torch.load(pkl_filename)
            logger.debug("samples shape: {}".format(samples.shape))

            logger.info("samples min-max: {}, {}".format(samples.min(), samples.max()))
            # print("skew min-max: {}, {}".format(skew.min(), skew.max()))
            # print("var min-max: {}, {}".format(skew.min(), skew.max()))

            plot_samples_grid(
                # TODO
                torch.clamp(samples[0:16], -1, 1),
                outfile=os.path.join(
                    args.savedir, "samples{}.{}.png".format(postfix, args.checkpoint)
                ),
                figsize=(8, 8)
                # title=str(dict(epoch=ep+1, var=best_var))
            )

            x_train = train_dataset.dataset.x_train
            mean_train_set = x_train.mean(dim=0, keepdim=True)

            mean_sample_set = (
                torch.clamp(samples, -1, 1).mean(dim=0, keepdim=True).detach().cpu()
            )
            print(
                "min max of mean train set: {:.3f}, {:.3f}".format(
                    mean_train_set.min(), mean_train_set.max()
                )
            )
            print(
                "min max of mean sample set: {:.3f}, {:.3f}".format(
                    mean_sample_set.min(), mean_sample_set.max()
                )
            )

            mean_samples = torch.cat(
                (
                    mean_train_set,
                    mean_sample_set,
                ),
                dim=0,
            )
            plot_samples(
                mean_samples,  # of shape (2, res, res, 2)
                subtitles=[
                    format_tuple(mean_train_set.min().item(), mean_train_set.max().item()),
                    format_tuple(
                        mean_sample_set.min().item(), mean_sample_set.max().item()
                    ),
                ],
                outfile=os.path.join(
                    args.savedir, "mean_sample{}.{}.png".format(postfix, args.checkpoint)
                ),
                figsize=(8, 4),
            )

            # import pdb; pdb.set_trace()

    else:

        raise ValueError("args.mode={} not recognised".format(args.mode))
