"""
Functions for the denoising Algorithmus
Note, conda and pip install of pycwt do not work correctly. Install pycwt directly from GitHub
"""

# Import
import copy
import numpy as np
from scipy import signal
from pycwt import pycwt


def langston_mousavi(x, noise, dt=1.0,
                     fmin=None, fmax=None, num=100,
                     wavelet="morlet", soft=True,
                     mode="empirical", quantile=0.99):
    """
    CWT denoising via thresholding functions. Array x is denoised via thresholding function
    that is definded by noise.
    The threshold function is applied between fmin and fmax. If both are None, then the whole CWT
    is denoised.

    References:
    For Details see "Separating Signal from Noise and from Other Signal
    Using Nonlinear Thresholding and Scale-Time Windowing of Continuous Wavelet Transforms" by
    Charles A. Langston and Seyed Mostafa Mousavi, BSSA, 2019

    :param x: Contains real data that will be denoised
    :type x: either np.array or list
    :param noise: Contains data that are used to build the threshold fucntion
    :type: either np.array or list
    :param dt: Sampling rate in s
    :param fmin: Minimum frequency for range where thresholding function is applied at CWT. Default is None
    :param fmax: Maximum frequency for range where thresholding function is applied at CWT. Default is None.
    :param num: Length of scales. If num is large, then denoising is accurate but takes a while.
    :param wavelet: instance of Wavelet class, or string. Mother wavelet class. Default is Morlet wavelet.
    :param soft: If True, soft thresholding is applied, otherwise hard thresholding
    :param mode: Mode for thresholding function. Either empirical or gaussian. Default is empirical
    :param quantile: Quantile of ECDF/Quantile-function to define threshold. Just requiered whne mode is empirical.
                     Default is 0.99

    :return: Real valued denoised numpy array
    """

    # Check function arguments
    if mode.lower() not in ["empirical", "gaussian"]:
        msg = "Argument mode must be either 'empirical' or 'gaussian' and not {}".format(mode)
        raise ValueError(msg)

    # Remove mean from x and noise
    x = x - np.mean(x)
    noise = noise - np.mean(noise)

    # Frequencies for CWT with numpy logspace
    freqs = np.logspace(start=np.log10(dt), stop=np.log10(1 / (2 * dt)), num=num)

    # Transforming x to TF-doamin
    cwt_x, scales_x, freqs_x, _, _, _ = pycwt.cwt(x, dt=dt, freqs=freqs, wavelet=wavelet)

    # Transforming noise to TF-domain
    cwt_n, scales_n, freq_n, _, _, _ = pycwt.cwt(noise, dt=dt, freqs=freqs, wavelet=wavelet)

    # Find indices for fmin and fmax
    if fmin is None or fmin <= 0.0:
        jmin = 0
    else:
        jmin = np.where(freqs_x > fmin)[0][0]

    if fmax is None or fmax >= 1 / (2*dt):
        jmax = num
    else:
        jmax = np.where(freqs_x < fmax)[0][-1]

    # Compute modified CWT
    cwt_abs = np.abs(cwt_x)
    cwt_xhat = copy.copy(cwt_x)          # Use copy instead of emtpy array to keep values that are not modified
    if mode.lower() == "empirical":      # by thresholding function
        # Loop over each scale
        for i in range(jmin, jmax):
            beta = np.quantile(cwt_n[i, :], q=quantile)       # Define thresholding function via ECDF (eq.14)
            # Loop over all times
            for j in range(cwt_x.shape[1]):
                if soft:                                       # (eq. 7)
                    if cwt_abs[i, j] >= beta:
                        cwt_xhat[i, j] = cwt_x[i, j] / cwt_abs[i, j] * (cwt_abs[i, j] - beta)
                    else:
                        cwt_xhat[i, j] = 0
                else:                                          # (eq. 6)
                    if cwt_abs[i, j] >= beta:
                        cwt_xhat[i, j] = cwt_x[i, j]
                    else:
                        cwt_xhat[i, j] = 0
    elif mode.lower() == "gaussian":
        # Define Donoho's universial threshold (eq. 12)
        c = np.sqrt(np.log10(len(x)))
        # Loop over all times
        for i in range(jmin, jmax):
            beta = np.mean(cwt_abs[i, :]) + c * np.std(cwt_abs[i, :])    # Thresholding function from universal
            # Loop over scales                                           # threshold (eq. 9)
            for j in range(cwt_x.shape[1]):
                if soft:
                    if cwt_abs[i, j] >= beta:
                        cwt_xhat[i, j] = cwt_x[i, j] / cwt_abs[i, j] * (cwt_abs[i, j] - beta)
                    else:
                        cwt_xhat[i, j] = 0
                else:
                    if cwt_abs[i, j] >= beta:
                        cwt_xhat[i, j] = cwt_x[i, j]
                    else:
                        cwt_xhat[i, j] = 0

    # Compute Inverse CWT
    # 1. Get correct dj to preserve energy (necessary for orthogonal wavelets)
    # As eq. (9) & (10) ind Torrence & Compo
    dj = 1 / num * np.log2(len(x) * dt / np.min(scales_x))     # Necessary because freqs are used for CWT

    # Perform ICWT
    x_hat = pycwt.icwt(cwt_xhat, sj=scales_x, dt=dt, dj=dj, wavelet=wavelet)

    return x_hat.real



def stft_thresholding(x, noise, dt=1.0, quantile=0.99,
                      fmin=None, fmax=None, soft=True,
                      **kwargs):
    """
    Denoising via quantile thresholding in fourier domain. First, array x is transformed via STFT into time-
    frequency-domain. Second, a threshold function is computed in fourier domain from noise. The spectrogram
    of x is modified by this threshold function and transformed back into time domain.

    :param x: Contains real data that will be denoised
    :type x: either np.array or list
    :param noise: Contains data that are used to build the threshold fucntion
    :type: either np.array or list
    :param dt: Sampling rate in s
    :param fmin: Minimum frequency for range where thresholding function is applied at CWT. Default is None
    :param fmax: Maximum frequency for range where thresholding function is applied at CWT. Default is None.
    :param soft: If True, soft thresholding is applied, otherwise hard thresholding
    :param quantile: Quantile of ECDF/Quantile-function to define threshold. Just requiered whne mode is empirical.
                     Default is 0.99
    :param kwargs: keyword arguments for short-time-fourier-transform. For further details see description of
                   scipy.signal.stft and scipy.signal.istft

    :return: Real valued denoised numpy array, Note, the length of returned array might differ from length of
             original array x.
    """

    # Remove mean from x and noise
    x = x - np.mean(x)
    noise = noise - np.mean(noise)

    # Get Time-Frequency-Representations from noisy signal and noise model
    fx, tx, Sx = signal.stft(x, fs=1/dt, **kwargs)
    fn, tn, Sn = signal.stft(noise, fs=1/dt, **kwargs)

    # Find indices for fmin and fmax
    if fmin is None or fmin <= 0.0:
        jmin = 0
    else:
        jmin = np.where(fx > fmin)[0][0]

    if fmax is None or fmax >= 1 / (2*dt):
        jmax = fx.shape[0]
    else:
        jmax = np.where(fx < fmax)[0][-1]

    # Compute modified spectrogram
    S_abs = np.abs(Sx)
    S_xhat = copy.copy(Sx)          # Use copy instead of emtpy array to keep values that are not modified
    # Loop over each scale
    for i in range(jmin, jmax):
        beta = np.quantile(Sn[i, :], q=quantile)       # Define thresholding function via Quantile
        # Loop over all times
        for j in range(Sx.shape[1]):
            if soft:
                if S_abs[i, j] >= beta:
                    S_xhat[i, j] = Sx[i, j] / S_abs[i, j] * (S_abs[i, j] - beta)
                else:
                    S_xhat[i, j] = 0
            else:
                if S_abs[i, j] >= beta:
                    S_xhat[i, j] = Sx[i, j]
                else:
                    S_xhat[i, j] = 0

    # Inverse STFT
    _, xhat = signal.istft(S_xhat, fs=1/dt, **kwargs)   # Length of x_hat might be unequal with length of x

    return xhat


def skimage_denoising(x, **kwargs):
    """
    Wrapper for denoising algorithm of skimage. For details of kwargs see description in skimage

    :param x: data for denoising
    :param kwargs:
    :return:
    """
    from skimage.restoration import denoise_wavelet
    # XXX Does perform badly in comparison to cwt denoising
    return denoise_wavelet(image=x, **kwargs)
