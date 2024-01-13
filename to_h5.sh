#!/bin/bash

#SBATCH --job-name=combine_channels
#SBATCH --time=00:30:00
#SBATCH --output=combine_channels_output.txt
#SBATCH --ntasks=1
#SBATCH --mem=1000
#SBATCH --exclude=node[066,067,115]
#SBATCH -c16
#SBATCH -p dicarlo
#SBATCH --mail-type=ALL


source /om2/user/sgouldin/miniconda3/etc/profile.d/conda.sh
conda activate brainio_dicarlo_pipeline

cd /braintree/data2/active/users/sgouldin/spike-tools-chong
python spike_tools/spike_proc.py -e combine_channels ${*:1}
EOF

# Submit a separate job to run spike_proc.py script with an additional argument, after mwk_ultra.py is completed
sbatch --job-name=combine_sessions --dependency=afterok:${SLURM_JOB_ID} --time=00:30:00 --output=/home/sgouldin/spike_tools/combine_sessions_output.txt --ntasks=1 --mem=1000 --exclude=node[066,067,115] -c16 -p dicarlo --mail-type=ALL <<EOF
#!/bin/bash

source /om2/user/sgouldin/miniconda3/etc/profile.d/conda.sh
conda activate brainio_dicarlo_pipeline

cd /braintree/data2/active/users/sgouldin/spike-tools-chong
python spike_tools/spike_proc.py -e combine_sessions ${*:1}
EOF

