#!/bin/bash

#SBATCH --job-name=mwk_ultra
#SBATCH --time=10:00:00
#SBATCH --output=/home/sgouldin/spike_tools/mwk_ultra_output.txt
#SBATCH --ntasks=1
#SBATCH --mem=10000
#SBATCH --exclude=node[066,067,115]
#SBATCH -c4
#SBATCH -p dicarlo
#SBATCH --mail-type=ALL

source /om2/user/sgouldin/miniconda3/etc/profile.d/conda.sh
conda activate brainio_dicarlo_pipeline

cd /braintree/data2/active/users/sgouldin/spike-tools-chong
python spike_tools/mwk_ultra.py ${*:1}
