#!/bin/bash
#SBATCH --partition=long                                 # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:32gb:1                                # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 16 GB of RAM
#SBATCH --time=0-16:00:00                                # The job will run for 16 hrs
#SBATCH -o /network/scratch/b/beckhamc/slurm-logs/slurm-cifar10-%j.out  # Write the log on scratch

cd ..

METHOD=$1
EXP_NAME=$2

if [[ $METHOD == "gan" ]]; then
  eval_file="eval_gano.py"
elif [[ $METHOD == "sbgm" ]]; then
  eval_file="eval.py"
else
  echo "Unknown method: choose either 'sbgm' or 'gan'"
  exit 1
fi
echo "eval file: " ${eval_file}

python ${eval_file} \
--exp_name=${SAVEDIR}/${EXP_NAME} \
--Ntest=1024 \
--val_batch_size=128 \
--savedir="${SAVEDIR}/${EXP_NAME}/eval" \
"${@:3}"
