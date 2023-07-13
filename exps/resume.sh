#!/bin/bash

# USAGE: sbatch resume.sh <exp name> <target exp>
# <exp name> can be whatever, and it is required
# <target_exp> is the experiment you WISH to resume
# e.g.
# sbatch resume.sh "resume_exp_9999" "my_existing_experiment/123456"

METHOD=$1
EXP_GROUP=$2
TARGET_EXP=$3

if [ $METHOD == "gan" ]; then
  resume_file="resume_gano.py"
elif [ $METHOD == "sbgm" ]; then
  resume_file="resume.py"
else
  echo "Unknown method: choose either 'sbgm' or 'gan'"
  exit 1
fi
echo "resume file: " ${resume_file}

if [ -z $EXP_GROUP ]; then
  echo "Must specify an experiment group name!"
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

if [ -z $RUN_LOCAL ]; then
  cd ${SAVEDIR}/${EXP_NAME}/code
else
  echo "RUN_LOCAL mode set, run code from this directory..."
fi
echo "Current working directory: " `pwd`

echo "Executing: python resume.py ${SAVEDIR}/${TARGET_EXP}"

python ${resume_file} $SAVEDIR/$TARGET_EXP

#bash launch.sh $EXP_NAME
