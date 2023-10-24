"""
This script check if signal and noise files are corrupted. Corrupted files will be deleted from hard drive.
To run the script type 'python check_dataset.py model_parfile
"""
import os.path
import sys

from utils import check_noise_files, check_signal_files, readtxt


# TODO: Add argparser instead of sys.argv and add flags to check either signal, noise or both

# Check input from command line
if len(sys.argv) <= 1:
    msg = "Parameter file is missing as argument. Run the script as 'python check_dataset.py model_parfile'."
    raise IOError(msg)
elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
    msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
    raise FileNotFoundError(msg)
else:
    parfile = sys.argv[1]

# Read settings
parameters = readtxt(fname=parfile)

# TODO: Replace .split() by something nicer
# Check signal files
check_signal_files(signal_dir=parameters['signal_pathname'].split("*")[0], verbose=True)

# Check noise files
check_noise_files(noise_dirname=parameters['noise_pathname'].split("*")[0], verbose=True)
