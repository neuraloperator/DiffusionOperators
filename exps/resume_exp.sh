#!/bin/bash
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:32gb:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 16 GB of RAM
#SBATCH --time=5-00:00:00                                   # The job will run for 4 days
#SBATCH -o /network/scratch/b/beckhamc/slurm-logs/slurm-cifar10-%j.out  # Write the log on scratch

if [ -z $RUN_LOCAL ]; then
  echo "Assuming sbatch experiment, automatically sourcing env.sh ..."
  source env.sh
fi

bash resume.sh "$@"
