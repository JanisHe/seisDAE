#!/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/geophysik/Schreibtisch/DeepDenoiser/"

conda activate py36
python model.py
