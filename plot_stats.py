from ast import parse
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.stats import wasserstein_distance as w_distance

from typing import Callable, Dict, Tuple


def plot_experiment_stats(
    exp_name: str,
    stats_filename: str,
    out_filename: str,
    gt_stats_filename: str = "gt_stats.pkl",
    figsize: Tuple[float, float] = (8, 4),
    title_fn: Callable = None,
    subtitle_fontsize: int = 14,
    title_fontsize: int = 14,
    verbose: bool = True,
) -> Dict[str, float]:
    with open(exp_name + "/{}".format(gt_stats_filename), "rb") as f:
        gt_stats = pickle.load(f)
    with open(exp_name + "/" + stats_filename, "rb") as f:
        stats = pickle.load(f)
    fig, axes = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
    buf = {}
    for j, varname in zip([0, 1], ["var", "skew"]):
        if varname == "var":
            xstart, xend = 0, 1
        else:
            xstart, xend = -4, 4
        # if verbose:
        #    print("length of generated stats: {}".format(len(stats[varname])))
        axes[j].hist(
            gt_stats[varname],
            bins=np.linspace(xstart, xend, 50),
            histtype="step",
            density=True,
            color="#ff7f0e",
            label="Ground truth",
        )
        axes[j].hist(
            stats[varname].numpy(),
            bins=np.linspace(xstart, xend, 50),
            color="#1f77b4",
            histtype="step",
            density=True,
            label="Model",
        )
        axes[j].legend()
        axes[j].set_xlim(xstart, xend)
        # ax[1].plot(lags_ref, acf_ref, c='r')
        # ax[1].plot(lags, acf, c='k')
        axes[j].set_xlabel("value")
        # plt.title('Histogram')
        axes[j].set_ylabel("count")
        buf[varname] = w_distance(gt_stats[varname], stats[varname].numpy())
        axes[j].set_title(
            "circular {} (w = {:.4f})".format(
                varname,
                buf[varname],
            )
        )
        axes[j].title.set_size(subtitle_fontsize)
        if verbose:
            print(varname, w_distance(gt_stats[varname], stats[varname].numpy()))
        # title_buf[varname] = w_distance(gt_stats[varname], stats[varname].numpy())
    if title_fn is not None:
        fig.suptitle(title_fn(exp_name, buf), fontsize=title_fontsize)
    fig.savefig(out_filename)
    return buf

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Full path of the experiment: <savedir>/<group>/<id>",
    )
    parser.add_argument("--stats_filename", type=str, required=True)
    parser.add_argument("--gt_stats_filename", type=str, default="gt_stats.pkl")
    parser.add_argument("--figsize", type=float, nargs='+', default=[8,4])
    parser.add_argument("--out_filename", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if len(args.figsize) != 2:
        raise ValueError("figsize must be a 2-tuple")
    plot_experiment_stats(
        exp_name=args.exp_name,
        stats_filename=args.stats_filename,
        gt_stats_filename=args.gt_stats_filename,
        out_filename=args.out_filename,
        figsize=args.figsize,
    )