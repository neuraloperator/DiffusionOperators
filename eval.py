import sys
import torch
import os
import json
import argparse
from utils import DotDict, circular_skew, circular_var

from train import init_model, sample

def parse_args():
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Full path of the experiment: <savedir>/<group>/<id>")
    parser.add_argument("--savedir", type=str, required=True,
                        help="dump stats here")
    parser.add_argument("--val_batch_size", type=int, default=512,
                        help="Batch size used for generating samples at inference time")
    parser.add_argument("--Ntest", type=int, default=1024,
                        help="Number of examples to generate for validation " + \
                            "(generating skew and variance metrics)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = DotDict(vars(parse_args()))
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    expdir = args.exp_name
    cfg = DotDict(
        json.loads(
            open(os.path.join(expdir, "config.json"),"r").read()
        )
    )
    #print(cfg)

    print(json.dumps(cfg, indent=4))

    fno, start_epoch, (train_dataset, valid_dataset), (init_sampler, noise_sampler, sigma) = \
        init_model(cfg)

    u, fn_outs = sample(
        fno, init_sampler, noise_sampler, sigma, 
        bs=args.val_batch_size, n_examples=args.Ntest, T=cfg.T,
        epsilon=cfg.epsilon,
        fns={"skew": circular_skew, "var": circular_var}
    )
    skew_generated = fn_outs['skew']
    var_generated = fn_outs['var']

    torch.save(
        (u.cpu(), skew_generated, var_generated),
        os.path.join(args.savedir, "samples.pkl")
    )
    