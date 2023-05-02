import sys
import torch
import os
import json
import argparse
import numpy as np
import pickle
from utils import (
    DotDict,
    circular_skew,
    circular_var,
    plot_samples,
    plot_samples_grid,
    min_max_norm,
    format_tuple,
    rescale,
    to_phase,
)

from train import init_model, sample, Arguments

from omegaconf import OmegaConf as OC

from setup_logger import get_logger

logger = get_logger(__name__)


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
        choices=["generate"],
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
    cfg = json.loads(open(os.path.join(expdir, "config.json"), "r").read())
    chkpt_file = os.path.join(expdir, args.checkpoint)
    # Validate the arguments. If anything is wrong this will
    # raise an exception.
    _ = OC.structured(Arguments(**cfg))
    cfg = DotDict(cfg)

    logger.info(json.dumps(cfg, indent=4))

    (
        G,
        D,
        ema_helper,
        start_epoch,
        (train_dataset, valid_dataset),
        noise_sampler,
    ) = init_model(cfg, args.savedir, checkpoint=chkpt_file)

    if args.mode == "generate":

        # Generate samples and dump them out to <savedir>/samples.pkl.

        with ema_helper:
            u, fn_outs = sample(
                G,
                noise_sampler,
                bs=args.val_batch_size,
                n_examples=args.Ntest,
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

        # elif args.mode == 'plot':

        samples = u.cpu()

        # samples, skew_generated, var_generated = torch.load(
        #    os.path.join(args.savedir, "samples.{}.pkl".format(args.checkpoint))
        # )

        logger.info("samples min-max: {}, {}".format(samples.min(), samples.max()))
        # print("skew min-max: {}, {}".format(skew.min(), skew.max()))
        # print("var min-max: {}, {}".format(skew.min(), skew.max()))

        plot_samples_grid(
            # TODO
            torch.clamp(samples[0:16], -1, 1),
            outfile=os.path.join(
                args.savedir, "samples.{}.png".format(args.checkpoint)
            ),
            figsize=(8, 8)
            # title=str(dict(epoch=ep+1, var=best_var))
        )

        x_train = train_dataset.dataset.x_train
        mean_train_set = x_train.mean(dim=0, keepdim=True)
        mean_sample_set = samples.mean(dim=0, keepdim=True).detach().cpu()
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
                args.savedir, "mean_sample.{}.png".format(args.checkpoint)
            ),
            figsize=(8, 4),
        )

        l2_mean_error = torch.sqrt(torch.sum((mean_train_set - mean_sample_set) ** 2))
        logger.info(
            "L2 between mean real vs mean sample image: {:.3f}".format(
                l2_mean_error
            )
        )
        l2_norm_mean = torch.sqrt(torch.sum(mean_sample_set**2))
        logger.info(
            "L2 norm for mean sample image: {:.3f}".format(l2_norm_mean)
        )

        l2_mean_error_p = torch.sqrt(torch.sum(
            (to_phase(mean_train_set) - to_phase(mean_sample_set)) ** 2
        ))
        logger.info(
            "L2 between mean real vs mean sample image (phase space): {:.3f}".format(
                l2_mean_error_p
            )
        )
        l2_norm_mean_p = torch.sqrt(torch.sum(to_phase(mean_sample_set) ** 2))
        logger.info(
            "L2 norm for mean sample image (phase space): {:.3f}".format(
                l2_norm_mean_p
            )
        )

        # import pdb; pdb.set_trace()

    else:

        raise ValueError("args.mode={} not recognised".format(args.mode))
