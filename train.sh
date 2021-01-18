#!/usr/bin/env bash

#qsub -l "low,h_vmem=5G,memory=5G" -cwd -hard -l os=*stretch -pe smp 80 train.sh


export PYTHONPATH="$PYTHONPATH:/home/janis/CODE/DeepDenoiser/"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate deep

python model.py