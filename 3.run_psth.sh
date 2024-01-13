#!/bin/bash

#SBATCH --job-name=psth
#SBATCH --array=0-191
#SBATCH --time=01:00:00
#SBATCH --output=psth.txt
#SBATCH --ntasks=1
#SBATCH --mem=1000
#SBATCH --exclude=node[066,067,115]
#SBATCH -c4
#SBATCH -p dicarlo
#SBATCH --mail-type=ALL

source /om2/user/sgouldin/miniconda3/etc/profile.d/conda.sh
conda activate brainio_dicarlo_pipeline

cd /braintree/data2/active/users/sgouldin/spike-tools-chong

# Run spike_proc.py as an array job
python spike_tools/spike_proc.py $SLURM_ARRAY_TASK_ID -e psth ${*:1}

