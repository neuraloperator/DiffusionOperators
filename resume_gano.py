import json
import sys
import os
from train_gano import run, Arguments
from src.util.utils import DotDict

from omegaconf import OmegaConf as OC

if __name__ == '__main__':

    exp_dir = sys.argv[1]
    args = json.loads(
        open(os.path.join(exp_dir, "config.json"), "r").read() 
    )
    # structured() allows type checking
    conf = OC.structured(Arguments(**args))

    # Since type checking is already done, convert
    # it back ito a (dot-accessible) dictionary.
    # (OC.to_object() returns back an Arguments object)
    run(OC.to_object(conf), exp_dir)
