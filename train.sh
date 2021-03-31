#!/usr/bin/env bash

# qsub -l "low,h_vmem=7.5G,memory=7.5G" -cwd -hard -q "low.q@minos15" -pe smp 30 train.sh model_parfile

export PYTHONPATH="$PYTHONPATH:/home/janis/CODE/cwt_denoiser/"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate py38

python run_model_from_parfile.py $1
