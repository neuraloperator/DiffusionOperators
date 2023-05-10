#!/bin/bash

EXP_GROUP=$1
CFG_FILE=$2

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
# it over. Then cd into that directory and run the code
if [ ! -d ${SAVEDIR}/${EXP_NAME}/code ]; then
  mkdir -p ${SAVEDIR}/${EXP_NAME}
  echo "Copying code..."
  rsync -r -v --exclude='exps' --exclude='.git' --exclude='__pycache__' --exclude '*.pyc' . ${SAVEDIR}/${EXP_NAME}/code
  if [ ! $? -eq 0 ]; then
    echo "rsync returned error, terminating..."
    exit 1
  fi
fi

CFG_ABS_PATH=`pwd`/exps/${CFG_FILE}
echo "Absolute path of cfg: " $CFG_ABS_PATH

if [ -z $RUN_LOCAL ]; then
  cd ${SAVEDIR}/${EXP_NAME}/code
  python train.py --cfg=$CFG_ABS_PATH --savedir=${SAVEDIR}/${EXP_NAME} --override_cfg
else
  echo "RUN_LOCAL mode set, run code from this directory..."
  python train.py --cfg=$CFG_ABS_PATH --savedir=${SAVEDIR}/${EXP_NAME}
fi
echo "Current working directory: " `pwd`

#bash launch.sh $EXP_NAME
