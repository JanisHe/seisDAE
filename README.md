# Denoising Autoencoder for seismic data denoising

This packages contains python script to train an neural network for denoising of seismological data.
Before starting, run the following command:
```
python setup.py install
```
There will be created a few new directories and the PyCWT packages is donloaded form GitHub,
which is necessary for computation of CWT.
The following packages should be installed in your Python environment:
 * Numpy
 * Matplotlib
 * Scipy
 * Tensorflow >= 2.0
 * Obspy 
 
After the installation of all packages, please check the file `train.sh` and change all
required parameters to run the script from your conda encrionment and in your directory.

Before starting the training, open the file `model_parfile` for the settings. The directory
`example_data` contains a small example dataset for signal and noise. 
The signal dataset are signals with high SNR from STEAD (Mousavi et al., 2020).

#### Start the Training
```
./train.sh model_parfile
```

#### Denoise data
Run the function `predict` from file `prediction.py` with your created model and
config file. The parameter data_list is a list with numpy arrays for denoising.

#### Denoise obspy Trace
Use the function `denoise_trace`in `denoise_day.py` and use the original trace.
your trained model and the conifg-file as input arguments. The denoising may take
some time and returns two traces for the recovered signal and noise.