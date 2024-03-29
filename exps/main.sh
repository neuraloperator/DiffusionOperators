#!/bin/bash

METHOD=$1
EXP_GROUP=$2
CFG_FILE=$3

if [[ $METHOD == "gan" ]]; then
  train_file="train_gano.py"
elif [[ $METHOD == "sbgm" ]]; then
  train_file="train.py"
else
  echo "Unknown method: choose either 'sbgm' or 'gan'"
  exit 1
fi
echo "train file: " ${train_file}

if [ -z $EXP_GROUP ]; then
  echo "Must specify an experiment group name!"
  exit 1
fi

if [ -z $CFG_FILE ]; then
  echo "Must specify a json file"
  exit 1
fi

if [ -z $SAVEDIR ]; then
  echo "SAVEDIR not found, source env.sh?"
  exit 1
fi

if [ -z "$SLURM_JOB_ID" ]; then
  echo "$SLURM_JOB_ID not set for some reason, are you in a login node?"
  echo "Set variable to '999999' for now"
  SLURM_JOB_ID=999999
fi

EXP_NAME="${EXP_GROUP}/${SLURM_JOB_ID}"
echo "Experiment name: " $EXP_NAME

cd ..

# If code does not exist for this experiment, copy
# it over. Then cd into that directory and run the code.
# But only if we're not in run_local mode.
if [ -z $RUN_LOCAL ]; then
  if [ ! -d ${SAVEDIR}/${EXP_NAME}/code ]; then
    mkdir -p ${SAVEDIR}/${EXP_NAME}
    echo "Copying code..."
    rsync -r -v --exclude='exps' --exclude='.git' --exclude='__pycache__' --exclude '*.pyc' . ${SAVEDIR}/${EXP_NAME}/code
    if [ ! $? -eq 0 ]; then
      echo "rsync returned error, terminating..."
      exit 1
    fi
  fi
fi

CFG_ABS_PATH=`pwd`/exps/${CFG_FILE}
echo "Absolute path of cfg: " $CFG_ABS_PATH
if [ -z $RUN_LOCAL ]; then
  cd ${SAVEDIR}/${EXP_NAME}/code
  python ${train_file} --cfg=$CFG_ABS_PATH --savedir=${SAVEDIR}/${EXP_NAME}
else
  echo "RUN_LOCAL mode set, run code from this directory..."
  # --override_cfg = use the local cfg file, not the one in the experiment directory
  python ${train_file} --cfg=$CFG_ABS_PATH --savedir=${SAVEDIR}/${EXP_NAME} --override_cfg
fi
echo "Current working directory: " `pwd`

#bash launch.sh $EXP_NAME
