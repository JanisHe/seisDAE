"""
Functions for prediction
"""

import random
import obspy

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import stft, istft

from pycwt import pycwt
from model import cwt_wrapper, Model, preprocessing
from utils import load_obj


def predict(model_filename, config_filename, data_list,  optimizer="adam", ckpt_model=True):
    """
    Function to predict data in data_list.

    Parameters:
        model_filename: Full filename of model or checkpoints
        config_filename: Full filename of config file
        data_list: List that contains numpy arrays for denoising
        optimizer: tensorflow optimizer for Model. Necessary if ckpt_model is True, default is adam.
        ckpt_model; True if tensorflow checkpoints are used as model. Set False if a full trained model is used, but
                    this is not necessary. Default is True

    Returns:
         recovered: Array that contains recoverd signal and noise. Has shape len(data_list)*ts_length*2, where
                    recovered[i, :, 0] denotes the recovered signal and recovered[i, :, 1] the recovered noise.
         transform_list: Array that contains all relevant transformations. Has shape
                         len(data_list)*shape_of_transformation*config['channels'] + 1.
                         First channel contains transformation of original signal, second channel contains
                         transformation of recovered signal and third channel of recovered noise.
    """

    # Read configs
    config = load_obj(config_filename)

    # Read tensorflow model
    if ckpt_model is True:
        # Load weights from checkpoint
        model_dae = Model(ts_length=config['ts_length'], use_bias=config['use_bias'],
                          activation=config['activation'], drop_rate=config['drop_rate'],
                          channels=config['channels'], optimizer=optimizer,
                          loss=config['loss'], dt=config['dt'], decimation_factor=config['decimation_factor'],
                          cwt=config['cwt'], **config['kwargs'])
        model_dae.build_model(depth=config['depth'], filter_root=config['filter_root'],
                              kernel_size=config['kernel_size'], strides=config['strides'],
                              fully_connected=config['fully_connected'])
        model_dae.model.load_weights(model_filename)
        input_shape = model_dae.shape
    else:
        # Read fully trained model
        model_dae = load_model(model_filename)
        input_shape = (model_dae.input_shape[1], model_dae.input_shape[2])

    # Allocate empty arrays for data
    X = np.empty(shape=(len(data_list), *input_shape, config['channels']), dtype="float")
    transform_list = np.empty(shape=(len(data_list), *input_shape, config['channels'] + 1), dtype="complex")
    scales = []
    dj = []
    norm_factors = []
    dt = config['dt']
    mean_values = []

    if config['channels'] == 1:
        phases = []

    # Loop over each data array in data_list, transform data and write into new array which is used for prediction
    for i, array in enumerate(data_list):
        signal_tmp = array[:config['ts_length']]

        signal, dt = preprocessing(data=signal_tmp, dt=config['dt'],
                                   decimation_factor=config['decimation_factor'])
                                   # taper=dict(max_percentage=0.02, type="cosine"),
                                   # filter=dict(type="highpass", freq=0.5))

        norm = np.max(np.abs(signal))
        signal = signal / norm
        norm_factors.append(norm)
        mean_values.append(np.mean(signal))

        # Transform data either using STFT of CWT
        if config['cwt'] is False:
            freqs, _, cns = stft(signal, fs=1 / dt, **config['kwargs'])
        elif config['cwt'] is True:
            cns, s, d_j, freqs = cwt_wrapper(signal, dt=dt, **config['kwargs'])
            scales.append(s)
            dj.append(d_j)

        # Allocate empty array for recovered signals
        if i == 0:
            recovered = np.empty(shape=(len(data_list), len(signal), 2), dtype="float")

        # Add transform to transform_list
        transform_list[i, :, :, 0] = cns

        # Write data to empty np arrays
        # X[i, :, :, 0] = np.abs(cns)
        X[i, :, :, 0] = cns.real / np.max(np.abs(cns.real))

        if config['channels'] == 1:
            phases.append(np.arctan2(cns.imag, cns.real))
        elif config['channels'] == 2:
            # X[i, :, :, 1] = np.arctan2(cns.imag, cns.real)
            X[i, :, :, 1] = cns.imag / np.max(np.abs(cns.imag))
        else:
            msg = "Channel number cannot exceed 2."
            raise ValueError(msg)

    # Denoise data by prediction with model
    if ckpt_model is True:
        X_pred = model_dae.model.predict(X)
    else:
        X_pred = model_dae.predict(X)

    # Loop over each element in predicted data and estimate denoised data
    for i in range(X_pred.shape[0]):
        if config['channels'] == 1:
            x_pred = X_pred[i, :, :, 0] * np.exp(1j * phases[i])
        elif config['channels'] == 2:
            # x_pred = transform_list[i, :, :, 0] * np.exp(1j * X_pred[i, :, :, 1])
            transform_list[i, :, :, 1] = transform_list[i, :, :, 0] * X_pred[i, :, :, 0]   # Recovered Signal
            transform_list[i, :, :, 2] = transform_list[i, :, :, 0] * X_pred[i, :, :, 1]   # Recovered Noise

        # Tranform modified transformation back into time-domain
        if config['cwt'] is False:
            t, rec_signal = istft(transform_list[i, :, :, 1], fs=1 / dt, **config['kwargs'])
            t, rec_noise = istft(transform_list[i, :, :, 2], fs=1 / dt, **config['kwargs'])
        elif config['cwt'] is True:
            rec_signal = pycwt.icwt(transform_list[i, :, :, 1], dt=dt, sj=scales[i], dj=dj[i])
            rec_noise = pycwt.icwt(transform_list[i, :, :, 2], dt=dt, sj=scales[i], dj=dj[i])

        # Multiply denoised trace by normalization factor to get true data without normalization
        rec_signal = np.real(rec_signal * norm_factors[i])
        rec_noise = np.real(rec_noise * norm_factors[i])

        # Add mean back on signal and noise
        rec_signal += mean_values[i]
        rec_noise += mean_values[i]

        # Append denoised traces to list
        recovered[i, :, 0] = rec_signal
        recovered[i, :, 1] = rec_noise

    return recovered, transform_list, freqs


def predict_test_dataset(model_filename, config_filename, signal_list, noise_list, optimizer="adam", ckpt_model=True):
    """
    Function to test a trained model on a test dataset.
    Plots all data and transformations.
    Parameters:
        model_filename: Full filename of model or checkpoints
        config_filename: Full filename of config file
        signal_list: List numpy array that contain Signals
        noise_list: List of numpy array that contain noise. Noise is added to signal to get noisy signal.
        optimizer: tensorflow optimizer for Model. Necessary if ckpt_model is True, default is adam.
        ckpt_model; True if tensorflow checkpoints are used as model. Set False if a full trained model is used, but
                    this is not necessary. Default is True

    Returns:

    """
    # Allocate empty list
    true_signal = []
    true_noise = []
    noisy_signal = []
    p_samp = []
    s_samp = []

    # Read config file
    config = load_obj(config_filename)

    # Loop over each signal in signal_list and create noisy signal
    for i, s in enumerate(signal_list):
        # Read signal
        signal = np.load(s)
        # XXX Read P- and S-arrival if available
        p_samp.append(signal["itp"])
        s_samp.append(signal["its"])

        signal = signal["data"][:config['ts_length']]

        # Read noise data
        noise = np.load("{}".format(noise_list[random.randint(0, len(noise_list) - 1)]))
        noise = noise["data"][:config['ts_length']]

        # Add noise and signal
        ns = signal + noise

        # Add noisy_signal and true signal to list
        true_signal.append(signal)
        true_noise.append(noise)
        noisy_signal.append(ns)

    # Denoise noisy signals
    recovered, transforms, _ = predict(model_filename, config_filename, noisy_signal, optimizer, ckpt_model)

    # Plot denoised signal and transformation
    for i in range(len(signal_list)):
        # Decimate signals for correct comparison
        if config['decimation_factor'] is not None:
            tr_ns = obspy.Trace(data=noisy_signal[i], header=dict(delta=config['dt']))
            tr_ns.decimate(factor=config['decimation_factor'])
            tr_ns.filter("highpass", freq=0.5)
            noisy_signal[i] = tr_ns.data

            tr_ts = obspy.Trace(data=true_signal[i], header=dict(delta=config['dt']))
            tr_ts.decimate(factor=config['decimation_factor'])
            tr_ts.filter("highpass", freq=0.5)
            true_signal[i] = tr_ts.data

            tr_n = obspy.Trace(data=true_noise[i], header=dict(delta=config['dt']))
            tr_n.decimate(factor=config['decimation_factor'])
            tr_n.filter("highpass", freq=0.5)
            true_noise[i] = tr_n.data

            dt = tr_ts.stats.delta
        else:
            dt = config['dt']

        # Define time axes
        t_max = len(noisy_signal[i]) * dt
        t_transform = np.linspace(0, t_max, num=transforms.shape[2])
        t_waveform = np.arange(0, len(noisy_signal[i])) * dt

        # Define frequency axis
        freq_max = 1 / (2 * dt)
        freqs = np.linspace(0, freq_max, num=transforms.shape[1])

        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.pcolormesh(t_transform, freqs, np.abs(transforms[i, :, :, 0]))

        ax3 = fig.add_subplot(323, sharex=ax1, sharey=ax1)
        ax3.pcolormesh(t_transform, freqs, np.abs(transforms[i, :, :, 1]))

        ax5 = fig.add_subplot(325, sharex=ax1, sharey=ax1)
        ax5.pcolormesh(t_transform, freqs, np.abs(transforms[i, :, :, 2]))

        ax2 = fig.add_subplot(322, sharex=ax1)
        ax2.plot(t_waveform, noisy_signal[i], color="k", alpha=0.5, label="Noisy Signal")
        plt.legend()

        ax4 = fig.add_subplot(324, sharex=ax1, sharey=ax2)
        ax4.plot(t_waveform, true_signal[i], alpha=0.5, color="k", label="True Signal")
        ax4.plot(t_waveform, recovered[i, :, 0], alpha=0.5, color="r", label="Denoised Signal")
        ax4.plot([p_samp[i]*config['dt'], p_samp[i]*config['dt']], [-1, 1], color="r")
        ax4.plot([s_samp[i]*config['dt'], s_samp[i]*config['dt']], [-1, 1], color="b")
        plt.legend()

        ax6 = fig.add_subplot(326, sharex=ax1, sharey=ax2)
        ax6.plot(t_waveform, true_noise[i], alpha=0.5, color="k", label="True Noise")
        ax6.plot(t_waveform, recovered[i, :, 1], alpha=0.5, color="r", label="Recovered Noise")
        ylim = max([np.max(np.abs(true_signal[i])), np.max(np.abs(recovered[i, :, 0])),
                    np.max(np.abs(recovered[i, :, 1])), np.max(np.abs(true_noise[i]))])
        ax6.set_ylim(-ylim, ylim)
        plt.legend()

        plt.show()


def test_model(model_filename, config_filename, **kwargs):
    import matplotlib.pyplot as plt
    # Create random time series
    config = load_obj(config_filename)
    data = np.random.normal(size=config['ts_length'])

    # Predict noise and signal
    recovered, _, _ = predict(model_filename=model_filename, config_filename=config_filename, data_list=[data],
                              **kwargs)

    # Apply preprocessing on data
    tr = obspy.Trace(data=data, header=dict(delta=config['dt']))
    tr.filter("highpass", freq=0.5)
    if config["decimation_factor"]:
        tr.decimate(factor=config["decimation_factor"])

    t_signal = np.arange(0, tr.stats.npts) * tr.stats.delta

    # Plot recovered data and compate recovered noise and signal to true data
    # If recovered noise + recovered signal do not match with true data, an error exists
    plt.plot(t_signal, tr.data, color="k", label="True Data", alpha=0.6)
    plt.plot(t_signal, recovered[0, :, 0] + recovered[0, :, 1], color="r", label="Recovered data", alpha=0.6)
    plt.legend()
    plt.show()




if __name__ == "__main__":
    import glob
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # from model import Model
    # signal_list = glob.glob("/home/geophysik/dae_noise_data/signal/*")[:10]
    # noise_list = glob.glob("/home/geophysik/dae_noise_data/noise/*/*/*")[:10]
    # signal_test_list = glob.glob("/home/geophysik/cwt_denoiser_test_data/*")
    #
    model = "/home/janis/CODE/cwt_denoiser/Models/test_cwt.h5"
    config = "/home/janis/CODE/cwt_denoiser/config/test_cwt.config"
    #
    # predict_test_dataset(model, config, signal_list, noise_list, ckpt_model=False)

    test_model(model_filename=model, config_filename=config)
