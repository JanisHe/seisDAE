from model import cwt_wrapper
from tensorflow.keras.models import load_model
import numpy as np
from pycwt import pycwt
import matplotlib.pyplot as plt
import copy
from scipy.signal import stft, istft
import glob
import obspy
import random
from pycwt import pycwt
import tensorflow as tf
from tensorflow import keras
from utils import load_obj


def cwt_wrapper(x, dt=1.0, yshape=150, **kwargs):

    # Remove mean from x
    x = x - np.mean(x)

    # Frequencies for CWT with numpy logspace
    #freqs = np.logspace(start=np.log10(dt), stop=np.log10(1 / (2 * dt)), num=yshape)
    freqs = np.linspace(dt, 1 / (2 * dt), yshape)

    # Transforming x to TF-doamin
    coeffs, scales, freqs_x, _, _, _ = pycwt.cwt(x, dt=dt, freqs=freqs, **kwargs)

    # Estimate dj as eq (9) & (10) in Torrence & Compo
    dj = 1 / yshape * np.log2(len(x) * dt / np.min(scales))

    return coeffs, scales, dj, freqs_x


def random_float(low, high):
    return random.random()*(high-low) + low


def predict(model_filename, config_filename, data_list, ckpt_model=False):

    # Read configs
    config = load_obj(config_filename)

    # Read tensorflow model
    if ckpt_model is True:
        # Load weights from checkpoint
        model = Model(ts_length=config['ts_length'], use_bias=config['use_bias'],
                      activation=config['activation'], drop_rate=config['drop_rate'],
                      channels=config['channels'], optimizer=config['optimizer'],
                      loss=config['loss'], dt=config['dt'], decimation_factor=config['decimation_factor'],
                      cwt=config['cwt'], **config['kwargs'])
        model.build_model(depth=config['depth'], filter_root=config['filter_root'], kernel_size=config['kernel_size'],
                          strides=config['strides'], fully_connected=config['fully_connected'])
        model.model.load_weights(model_filename)
    else:
        # Read fully trained model
        model = load_model(model_filename)


def prediction(model_pathname, signal_list, noise_list, nfft=128, nperseg=128, noverlap=16, cwt=False):
    model = load_model(model_pathname)
    X = np.empty(shape=(len(signal_list), *(model.input_shape[1], model.input_shape[2]), model.input_shape[3]),
                 dtype="float")
    channels = model.input_shape[3]
    x_noisy = []
    stft_list = []
    x_true = []

    if channels == 1:
        phases = []

    for i, signal in enumerate(signal_list):
        signal = np.load(signal)
        noise = np.load("{}".format(noise_list[random.randint(0, len(noise_list) - 1)]))
        signal = signal.f.data[:6001]  #, 0]
        noise = noise.f.data[:6001]  #, 0]

        # Remove mean
        noise = noise - np.mean(noise)
        signal = signal - np.mean(signal)

        # Apply highpass filter
        tr_n = obspy.Trace(data=noise, header=dict(delta=0.01))
        tr_s = obspy.Trace(data=signal, header=dict(delta=0.01))
        tr_n.filter("highpass", freq=0.5)
        tr_s.filter("highpass", freq=0.5)

        # Decimate by factor 2
        #tr_n.decimate(factor=2)
        #tr_s.decimate(factor=2)

        # Normalize Noise and signal
        noise = tr_n.data
        signal = tr_s.data
        noise = noise / np.max(np.abs(noise))
        signal = signal / np.max(np.abs(signal))

        # Adding signal and noise
        rand_noise = random_float(0, 2)
        rand_signal = random_float(0, 2)
        noisy_signal = rand_signal * signal + rand_noise * noise

        # Normalize Signal and Noise
        norm = np.max(np.abs(noisy_signal))
        noisy_signal = noisy_signal / norm
        x_noisy.append(noisy_signal)
        x_true.append(signal)

        if cwt is False:
            _, _, cns = stft(noisy_signal, fs=100, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        elif cwt is True:
            cns, scales, dj, freqs = cwt_wrapper(noisy_signal, dt=0.01, yshape=model.input_shape[1])
        stft_list.append(cns)

        # Write data to empty np arrays
        # X[i, :, :, 0] = np.abs(cns)
        X[i, :, :, 0] = cns.real

        if channels == 1:
            phases.append(np.arctan2(cns.imag, cns.real))
        elif channels == 2:
            # X[i, :, :, 1] = np.arctan2(cns.imag, cns.real)
            X[i, :, :, 1] = cns.imag
        else:
            msg = "Channel number cannot exceed 2."
            raise ValueError(msg)

    pred = model.predict(X)

    for i in range(pred.shape[0]):

        if channels == 1:
            x_pred = pred[i, :, :, 0] * np.exp(1j * phases[i])
            # x_pred = np.abs(stft_list[i]) * pred[i, :, :, 0] * np.exp(1j * phases[i])
        elif channels == 2:
            # x_pred = pred[i, :, :, 0] * np.exp(1j * pred[i, :, :, 1])
            x_pred = stft_list[i] * pred[i, :, :, 0]

        plt.figure()
        plt.subplot(221)
        #plt.pcolormesh(np.abs(X[i, :, :, 0]))
        plt.pcolormesh(np.abs(stft_list[i]))

        plt.subplot(223)
        plt.pcolormesh(np.abs(x_pred))

        if cwt is False:
            t, x_denoi = istft(x_pred, fs=100, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        elif cwt is True:
            x_denoi = pycwt.icwt(x_pred, dt=0.01, sj=scales, dj=dj)

        plt.subplot(222)
        plt.plot(x_noisy[i])

        plt.subplot(224)
        plt.plot(x_true[i], alpha=0.5, label="True Signal")
        plt.plot(x_denoi / np.max(np.abs(x_denoi)), alpha=0.5, label="Denoised")
        plt.legend()

        plt.show()




def predict_test_data(model, signal_list, nfft=128, nperseg=128, noverlap=16, cwt=False, decimation_factor=2):
    X = np.empty(shape=(len(signal_list), *(model.input_shape[1], model.input_shape[2]), model.input_shape[3]),
                 dtype="float")
    channels = model.input_shape[3]
    x_noisy = []
    stft_list = []
    names = []
    p_picks = []
    s_picks = []
    norms = []

    if channels == 1:
        phases = []

    for i, signal in enumerate(signal_list):
        names.append(signal)
        signal = np.load(signal)
        p_picks.append(np.ceil(signal.f.itp))
        s_picks.append(np.ceil(signal.f.its))
        signal = signal.f.data[:6001]  # , 0]

        # Remove mean
        signal = signal - np.mean(signal)

        # Apply highpass filter
        tr_s = obspy.Trace(data=signal, header=dict(delta=0.01))
        tr_s.filter("highpass", freq=0.5)
        tr_s.taper(0.02, type="cosine")

        # Decimate by factor 2
        tr_s.decimate(factor=decimation_factor)

        # Normalize Signal
        signal = tr_s.data
        norm = np.max(np.abs(signal))
        signal = signal / norm

        x_noisy.append(signal)
        norms.append(norm)

        if cwt is False:
            _, _, cns = stft(signal, fs=100, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        elif cwt is True:
            cns, scales, dj, freqs = cwt_wrapper(signal, dt=tr_s.stats.delta, yshape=model.input_shape[1])
        stft_list.append(cns)

        # Write data to empty np arrays
        # X[i, :, :, 0] = np.abs(cns)
        X[i, :, :, 0] = cns.real

        if channels == 1:
            phases.append(np.arctan2(cns.imag, cns.real))
        elif channels == 2:
            # X[i, :, :, 1] = np.arctan2(cns.imag, cns.real)
            X[i, :, :, 1] = cns.imag
        else:
            msg = "Channel number cannot exceed 2."
            raise ValueError(msg)

    pred = model.predict(X)

    for i in range(pred.shape[0]):

        if channels == 1:
            x_pred = pred[i, :, :, 0] * np.exp(1j * phases[i])
            # x_pred = np.abs(stft_list[i]) * pred[i, :, :, 0] * np.exp(1j * phases[i])
        elif channels == 2:
            # x_pred = pred[i, :, :, 0] * np.exp(1j * pred[i, :, :, 1])
            x_pred = stft_list[i] * pred[i, :, :, 0]

        if decimation_factor == 1 or decimation_factor is None:
            samples = np.arange(0, 6001)
        elif decimation_factor == 2:
            samples = np.arange(0, 3001)
        elif decimation_factor == 4:
            samples = np.arange(0, 1501)

        plt.figure()
        plt.suptitle(names[i])
        ax1 = plt.subplot(221)
        #plt.pcolormesh(np.abs(X[i, :, :, 0]))
        if cwt is True:
            plt.pcolormesh(samples, freqs, np.abs(stft_list[i]))
        plt.ylabel("Frequency (Hz)")

        plt.subplot(223, sharex=ax1, sharey=ax1)
        if cwt is True:
            plt.pcolormesh(samples, freqs, np.abs(x_pred))
        plt.xlabel("Samples")
        plt.ylabel("Frequency (Hz)")

        if cwt is False:
            t, x_denoi = istft(x_pred, fs=100, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        elif cwt is True:
            x_denoi = pycwt.icwt(x_pred, dt=tr_s.stats.delta, sj=scales, dj=dj)

        ax2 = plt.subplot(222, sharex=ax1)
        plt.plot(x_noisy[i] * norms[i], label="Original data", color="k")
        plt.plot([p_picks[i]/ decimation_factor, p_picks[i]/decimation_factor], [-1, 1], color="r", label="P")
        plt.plot([s_picks[i] / decimation_factor, s_picks[i] / decimation_factor], [-1, 1], color="b", label="S")
        plt.ylabel("Normalized Amplitude")
        plt.ylim([np.min(x_noisy[i] * norms[i]), np.max(x_noisy[i] * norms[i])])
        plt.legend()

        plt.subplot(224, sharex=ax1, sharey=ax2)
        plt.plot(x_denoi * norms[i], alpha=1, color="k", label="Denoised")
        plt.plot([p_picks[i] / decimation_factor, p_picks[i] / decimation_factor], [-1, 1], color="r", label="P")
        plt.plot([s_picks[i] / decimation_factor, s_picks[i] / decimation_factor], [-1, 1], color="b", label="S")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.legend()

        plt.show()



if __name__ == "__main__":
    import glob
    from model import Model
    signal_list = glob.glob("/home/geophysik/dae_noise_data/signal/*")[:10]
    noise_list = glob.glob("/home/geophysik/dae_noise_data/noise/*")[:10]
    signal_test_list = glob.glob("/home/geophysik/cwt_denoiser_test_data/*")

    # Load fully model
    model = "/home/geophysik/Schreibtisch/cwt_denoiser/Models/2021-01-08_cwt.h5"
    model = load_model(model)

    # # Load weights from checkpoint
    # # Prediction: model.model.predict
    # cpkt = "/home/geophysik/Schreibtisch/cwt_denoiser/checkpoints/latest_checkpoint.ckpt"
    # model = Model(ts_length=6001, use_bias=False, activation=None, drop_rate=0.001, channels=2, optimizer="adam",
    #               loss='mean_squared_error', dt=0.01, decimation_factor=2, cwt=True, yshape=80)
    # model.build_model(depth=6)
    # model.model.load_weights(cpkt)

    #prediction(model, signal_list, noise_list, cwt=False, nfft=61, noverlap=16, nperseg=31)
    predict_test_data(model, signal_test_list, cwt=True, decimation_factor=2)
