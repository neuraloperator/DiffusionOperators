import json
import sys
import os
from Volcano_train import run
from utils import DotDict

if __name__ == '__main__':

    exp_dir = sys.argv[1]
    args = json.loads( open(os.path.join(exp_dir, "config.json"), "r").read() )
    args = DotDict(args)
    args.savedir = exp_dir

    run(args)