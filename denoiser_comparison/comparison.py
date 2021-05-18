import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from denoiser_comparison.denoising import langston_mousavi
from prediction import predict
from model import cwt_wrapper
import obspy
import datetime


def rms(x):
    """
    Root mean square of array x
    :param x:
    :return:
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal, noise):
    """
    SNR in dB
    :param signal:
    :param noise:
    :return:
    """
    return 10 * np.log10(rms(signal) / rms(noise))


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Set pathnames for CWT and STFT Models and config files
model_filename_cwt = "/home/janis/CODE/cwt_denoiser/Models/BAVN_cwt_raw.h5"
config_filename_cwt = "/home/janis/CODE/cwt_denoiser/config/BAVN_cwt_raw.config"
model_filename_stft = "/home/janis/CODE/cwt_denoiser/Models/BAVN_stft_raw.h5"
config_filename_stft = "/home/janis/CODE/cwt_denoiser/config/BAVN_stft_raw.config"

# Read data for Denoising
# Note, the arrays must be saved as np.savez(filename, data=...)
signal_test_list = glob.glob("/rscratch/minos14/janis/cwt_denoiser_test_data/BAVN/all_events/*")[:]

# Some more settings
dt = 0.01
sample_length = 6001
filter_type = dict(type="bandpass", freqmin=5, freqmax=20)   # Comparison for filter -> see obspy filter methods!
colorbar = False     # If true, creates new figure just for colorbars to add later in inkscape

# END OF SETTINGS


# Allocate empty lists
signals = []
p_arrivals = []
s_arrivals = []
names = []
snr = []
starttimes = []

# Prepare data for DAE
for i, signal in enumerate(signal_test_list):
    names.append(signal)
    s = np.load(signal)
    signal_tmp = s.f.data[:sample_length]
    try:
        starttimes.append(s['starttime'])
    except KeyError:
        pass

    # Apply highpass filter to data
    tr_s = obspy.Trace(data=signal_tmp, header=dict(delta=dt))
    # tr_s.filter("highpass", freq=0.1)
    # tr_s.taper(0.02, type="cosine")

    # Add data to list
    signals.append(tr_s.data)

    # Read phase arrivals
    p_arrivals.append(s.f.itp * dt)
    s_arrivals.append(s.f.its * dt)

    # Compute SNR of p-arrival
    snr.append(signal_to_noise_ratio(signal=tr_s.data[int(s["itp"]-50):int(s["itp"]+250)],
                                     noise=tr_s.data[int(s["itp"]-400):int(s["itp"]-100)]))

# Denoise data by DAE
# 1. Using CWT
recovered_cwt, transforms_cwt, freqs_cwt = predict(model_filename_cwt, config_filename_cwt, signals, ckpt_model=True)
# 2. Using STFT
recovered_stft, transforms_stft, freqs_stft = predict(model_filename_stft, config_filename_stft, signals,
                                                      ckpt_model=True)

# Loop over all signals, denoise with threshold function and plot results
for i, signal in enumerate(signals):
    print(names[i])
    # Denoise with threshold function
    if int(p_arrivals[i] / dt) - 50 > 0:
        s_threshold = langston_mousavi(x=signal, noise=signal[:int(p_arrivals[i] / dt) - 100], dt=dt, num=150,
                                       freqs_log=True)
    else:
        # Take first five seconds as noise
        s_threshold = langston_mousavi(x=signal, noise=signal[:int(5 / dt)], dt=dt, num=150,
                                       freqs_log=True)
    n_threshold = signal - s_threshold

    # Bandpass filter
    tr_bandpass = obspy.Trace(data=signal, header=dict(delta=dt))
    tr_bandpass.filter(**filter_type)
    s_bandpass = tr_bandpass.data
    n_bandpass = signal - s_bandpass

    # CWT of s_threshold
    thresh_coeffs, scales, dj, freqs_thresh = cwt_wrapper(s_threshold, dt=dt, yshape=150)
    n_thresh_coeffs, _, _, n_freqs_thresh = cwt_wrapper(n_threshold, dt=dt, yshape=150)

    # CWT of bandpass
    thresh_coeffs_bandpass, _ , _, freqs_thresh_bandpass = cwt_wrapper(s_bandpass, dt=dt, yshape=150)
    n_thresh_coeffs_bandpass, _, _, n_freqs_thresh_bandpass = cwt_wrapper(n_bandpass, dt=dt, yshape=150)

    # Make time arrays for plot
    t_signal = np.arange(0, len(signal)) * dt
    t_dae_cwt = np.linspace(0, max(t_signal), num=transforms_cwt.shape[2])
    t_dae_stft = np.linspace(0, max(t_signal), num=transforms_stft.shape[2])
    t_dae_stft_waveform = np.linspace(0, max(t_signal), num=len(recovered_stft[i, :, 0]))

    freq_max = min([max(freqs_thresh), max(freqs_cwt), max(freqs_stft)])
    ampl_max = max([max(np.abs(recovered_cwt[i, :, 0])), max(np.abs(recovered_stft[i, :, 0])), max(np.abs(signal)),
                    max(np.abs(s_threshold))])

    # SNR of denoised signals
    dt_cwt = t_dae_cwt[1] - t_dae_cwt[0]
    decimation_cwt = dt_cwt / dt
    dt_stft = t_dae_stft_waveform[1] - t_dae_stft_waveform[0]
    decimation_stft = dt_stft / dt
    snr_cwt = signal_to_noise_ratio(signal=recovered_cwt[i, :, 0][int(p_arrivals[i]/dt_cwt - 50/decimation_cwt):
                                                                  int(p_arrivals[i]/dt_cwt + 250/decimation_cwt)],
                                    noise=recovered_cwt[i, :, 0][int(p_arrivals[i]/dt_cwt - 400/decimation_cwt):
                                                                 int(p_arrivals[i]/dt_cwt - 100/decimation_cwt)])
    snr_stft = signal_to_noise_ratio(signal=recovered_stft[i, :, 0][int(p_arrivals[i]/dt_stft - 50/decimation_stft):
                                                                    int(p_arrivals[i]/dt_stft + 250/decimation_stft)],
                                     noise=recovered_stft[i, :, 0][int(p_arrivals[i]/dt_stft - 400/decimation_stft):
                                                                   int(p_arrivals[i]/dt_stft - 100/decimation_stft)])
    snr_threshold = signal_to_noise_ratio(signal=s_threshold[int(p_arrivals[i]/dt_stft - 50/decimation_stft):
                                                             int(p_arrivals[i]/dt_stft + 250/decimation_stft)],
                                          noise=s_threshold[int(p_arrivals[i]/dt_stft - 400/decimation_stft):
                                                            int(p_arrivals[i]/dt_stft - 100/decimation_stft)])
    snr_bandpass = signal_to_noise_ratio(signal=s_bandpass[int(p_arrivals[i]/dt_stft - 50/decimation_stft):
                                                           int(p_arrivals[i]/dt_stft + 250/decimation_stft)],
                                         noise=s_bandpass[int(p_arrivals[i]/dt_stft - 400/decimation_stft):
                                                          int(p_arrivals[i]/dt_stft - 100/decimation_stft)])

    # Transform all time objects to datime objects to plot correct timestamps on x-axis
    # Comment out the following lines if absolute time is not needed!
    if len(starttimes) == len(signals):
        t_signal = [obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=k) for k in t_signal]
        t_dae_cwt = [obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=k) for k in t_dae_cwt]
        t_dae_stft = [obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=k) for k in t_dae_stft]
        t_dae_stft_waveform = [obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=k) for k in
                               t_dae_stft_waveform]

        p_arrivals[i] = obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=p_arrivals[i])
        s_arrivals[i] = obspy.UTCDateTime(str(starttimes[i])).datetime + datetime.timedelta(seconds=s_arrivals[i])

    # Plot Signal
    fig = plt.figure(figsize=(16, 8))
    if colorbar is True:
        fig_c = plt.figure(figsize=(4, 8))
    # fig.suptitle("SNR: {} | SNR {}{}: {} | SNR Thresh: {} | SNR CWT: {} | SNR STFT: {}".format(
    #     np.round(snr[i], 2),
    #     filter_type['type'][0].upper(),
    #     filter_type['type'][1:],
    #     np.round(snr_bandpass, 2),
    #     np.round(snr_threshold, 2),
    #     np.round(snr_cwt, 2),
    #     np.round(snr_stft, 2))
    # )

    # XXX Add colorbar for each subplot -> own colorbar
    vmin_cwt = np.quantile(np.abs(transforms_cwt[i, :, :, 0]).flatten(), 0.001)
    vmax_cwt = np.quantile(np.abs(transforms_cwt[i, :, :, 0]).flatten(), 0.999)

    # Original Signal
    ax1 = fig.add_subplot(522)
    pcolor = ax1.pcolormesh(t_dae_cwt, freqs_cwt, np.abs(transforms_cwt[i, :, :, 0]), shading="auto", rasterized=True)
    if colorbar is True:
        pcolor = ax1.pcolormesh(t_dae_cwt, freqs_cwt, np.abs(transforms_cwt[i, :, :, 0]), shading="auto",
                                rasterized=True, norm=colors.LogNorm(vmin_cwt, vmax_cwt))
        ax1_c = fig_c.add_subplot(511)
        cbar = fig_c.colorbar(pcolor, ax=ax1_c)
        cbar.set_label(r'|CWT(a, $\tau$)|')
    #ax1.title.set_text("Original Data")

    ax2 = fig.add_subplot(521, sharex=ax1)
    ax2.plot(t_signal, signal - np.mean(signal), color="k", rasterized=True)
    ax2.plot([p_arrivals[i], p_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="r", alpha=0.5)
    ax2.plot([s_arrivals[i], s_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="b", alpha=0.5)

    ax2.text(0.05, 0.95, "Raw data",
             verticalalignment='top', horizontalalignment='left',
             transform=ax2.transAxes,
             color='k', fontsize=12,
             )

    if np.isnan(snr[i]) == False:
        ax2.text(0.85, 0.95, "SNR={}".format(np.round(snr[i], 2)),
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax2.transAxes,
                 color='k', fontsize=12,
                 )

    # Bandpass filter
    ax3a = fig.add_subplot(524, sharex=ax1, sharey=ax1)
    ax3a.pcolormesh(t_signal, freqs_thresh_bandpass, np.abs(thresh_coeffs_bandpass), shading="auto", rasterized=True)
    title_bp = "{}{} Filter".format(filter_type['type'][0].upper(), filter_type['type'][1:].lower())
    if filter_type['type'] in ["bandpass", "bandstop"]:
        title_bp += " {}-{} Hz".format(filter_type['freqmin'], filter_type['freqmax'])
    else:
        title_bp += " {} Hz".format(filter_type['freq'])
    # ax3a.set_title(title_bp)

    ax4a = fig.add_subplot(523, sharex=ax1, sharey=ax2)
    ax4a.plot(t_signal, s_bandpass, color="k", rasterized=True)
    ax4a.plot([p_arrivals[i], p_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="r", alpha=0.5)
    ax4a.plot([s_arrivals[i], s_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="b", alpha=0.5)

    ax4a.text(0.05, 0.95, title_bp,
              verticalalignment='top', horizontalalignment='left',
              transform=ax4a.transAxes,
              color='k', fontsize=12,
              )

    if np.isnan(snr_bandpass) == False:
        ax4a.text(0.85, 0.95, "SNR={}".format(np.round(snr_bandpass, 2)),
                  verticalalignment='top', horizontalalignment='left',
                  transform=ax4a.transAxes,
                  color='k', fontsize=12,
                  )

    # CWT Denoiser / Threshold function
    ax3 = fig.add_subplot(526, sharex=ax1, sharey=ax1)
    ax3.pcolormesh(t_signal, freqs_thresh, np.abs(thresh_coeffs), shading="auto", rasterized=True)
    # ax3.title.set_text("CWT Threshold Denoising")
    ax3.set_ylabel("Frequency (Hz)")

    ax4 = fig.add_subplot(525, sharex=ax1, sharey=ax2)
    ax4.plot(t_signal, s_threshold, color="k", rasterized=True)
    ax4.plot([p_arrivals[i], p_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="r", alpha=0.5)
    ax4.plot([s_arrivals[i], s_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="b", alpha=0.5)
    ax4.set_ylabel("Amplitude (counts)")  # (m/s)")
    ax4.text(0.05, 0.95, "CWT Thresholding",
             verticalalignment='top', horizontalalignment='left',
             transform=ax4.transAxes,
             color='k', fontsize=12,
             )

    if np.isnan(snr_threshold) == False:
        ax4.text(0.85, 0.95, "SNR={}".format(np.round(snr_threshold, 2)),
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax4.transAxes,
                 color='k', fontsize=12,
                 )

    # DAE with CWT
    ax5 = fig.add_subplot(528, sharex=ax1, sharey=ax1)
    ax5.pcolormesh(t_dae_cwt, freqs_cwt, np.abs(transforms_cwt[i, :, :, 1]), shading="auto", rasterized=True)
    # ax5.title.set_text("DAE CWT")

    ax6 = fig.add_subplot(527, sharex=ax1, sharey=ax2)
    ax6.plot(t_dae_cwt, recovered_cwt[i, :, 0], color="k")
    ax6.plot([p_arrivals[i], p_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="r", alpha=0.5)
    ax6.plot([s_arrivals[i], s_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="b", alpha=0.5)
    ax6.text(0.05, 0.95, "DAE CWT",
             verticalalignment='top', horizontalalignment='left',
             transform=ax6.transAxes,
             color='k', fontsize=12,
             )

    if np.isnan(snr_cwt) == False:
        ax6.text(0.85, 0.95, "SNR={}".format(np.round(snr_cwt, 2)),
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax6.transAxes,
                 color='k', fontsize=12,
                 )

    # DAE with STFT
    ax7 = fig.add_subplot(5, 2, 10, sharex=ax1, sharey=ax1)
    ax7.pcolormesh(t_dae_stft, freqs_stft, np.abs(transforms_stft[i, :, :, 1]), shading="auto", rasterized=True)
    ax7.set_ylim(0, freq_max)
    # ax7.title.set_text("DAE STFT")
    if len(starttimes) == len(signals):
        ax7.set_xlabel("Time since {:04d}-{:02d}-{:02d} 00:00:00".format(obspy.UTCDateTime(str(starttimes[i])).year,
                                                                         obspy.UTCDateTime(str(starttimes[i])).month,
                                                                         obspy.UTCDateTime(str(starttimes[i])).day))
    else:
        ax7.set_xlabel("Time (s)")

    ax8 = fig.add_subplot(5, 2, 9, sharex=ax1, sharey=ax2)
    ax8.plot(t_dae_stft_waveform, recovered_stft[i, :, 0], color="k")
    ax8.plot([p_arrivals[i], p_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="r", alpha=0.5)
    ax8.plot([s_arrivals[i], s_arrivals[i]], [-1 - ampl_max, 1 + ampl_max], color="b", alpha=0.5)
    ax8.set_ylim(-ampl_max, ampl_max)
    ax8.set_xlim(min(t_signal), max(t_signal))
    if len(starttimes) == len(signals):
        ax8.set_xlabel("Time since {:04d}-{:02d}-{:02d} 00:00:00".format(obspy.UTCDateTime(str(starttimes[i])).year,
                                                                         obspy.UTCDateTime(str(starttimes[i])).month,
                                                                         obspy.UTCDateTime(str(starttimes[i])).day))
    else:
        ax8.set_xlabel("Time (s)")
    ax8.text(0.05, 0.95, "DAE STFT",
             verticalalignment='top', horizontalalignment='left',
             transform=ax8.transAxes,
             color='k', fontsize=12,
             )
    fig.tight_layout()

    if np.isnan(snr_stft) == False:
        ax8.text(0.85, 0.95, "SNR={}".format(np.round(snr_stft, 2)),
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax8.transAxes,
                 color='k', fontsize=12,
                 )

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax3a, ax4a]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Figure zoom
    # fig_zoom = plt.figure()
    # az1 = fig_zoom.add_subplot(511)
    # az1.plot(t_signal, signal, color="k")
    #
    # az2 = fig_zoom.add_subplot(512, sharex=az1)
    # az2.plot(t_signal, s_bandpass, color="k")
    #
    # az3 = fig_zoom.add_subplot(513, sharex=az1)
    # az3.plot(t_signal, s_threshold, color="k")
    #
    # az4 = fig_zoom.add_subplot(514, sharex=az1)
    # az4.plot(t_dae_cwt, recovered_cwt[i, :, 0], color="k")
    #
    # az5 = fig_zoom.add_subplot(515, sharex=az1)
    # az5.plot(t_dae_stft_waveform, recovered_stft[i, :, 0], color="k")
    #
    # for ax in [az1, az2, az3, az4, az5]:
    #     plt.setp(ax.get_xticklabels(), visible=False)


    # Plot recovered noise
    fig_n = plt.figure(figsize=(16, 8))

    # Original Signal
    ax1 = fig_n.add_subplot(522)
    ax1.pcolormesh(t_dae_cwt, freqs_cwt, np.abs(transforms_cwt[i, :, :, 0]), shading="auto")
    ax1.title.set_text("Original Data")

    ax2 = fig_n.add_subplot(521, sharex=ax1)
    ax2.plot(t_signal, signal, color="k")
    ax2.plot(t_dae_cwt, recovered_cwt[i, :, 1] + recovered_cwt[i, :, 0], color="r")

    # Bandpass filter
    ax3a = fig_n.add_subplot(524, sharex=ax1, sharey=ax1)
    ax3a.pcolormesh(t_signal, freqs_thresh_bandpass, np.abs(n_thresh_coeffs_bandpass), shading="auto", rasterized=True)
    ax3a.set_title(title_bp)

    ax4a = fig_n.add_subplot(523, sharex=ax1, sharey=ax2)
    ax4a.plot(t_signal, n_bandpass, color="k", rasterized=True)

    # CWT Denoiser / Threshold function
    ax3 = fig_n.add_subplot(526, sharex=ax1, sharey=ax1)
    ax3.pcolormesh(t_signal, freqs_thresh, np.abs(n_thresh_coeffs), shading="auto")
    ax3.title.set_text("CWT Threshold Denoising")
    ax3.set_ylabel("Frequency (Hz)")

    ax4 = fig_n.add_subplot(525, sharex=ax1, sharey=ax2)
    ax4.plot(t_signal, n_threshold, color="k")
    ax4.set_ylabel("Amplitude (m/s)")

    # DAE with CWT
    ax5 = fig_n.add_subplot(528, sharex=ax1, sharey=ax1)
    ax5.pcolormesh(t_dae_cwt, freqs_cwt, np.abs(transforms_cwt[i, :, :, 2]), shading="auto")
    ax5.title.set_text("DAE CWT")

    ax6 = fig_n.add_subplot(527, sharex=ax1, sharey=ax2)
    ax6.plot(t_dae_cwt, recovered_cwt[i, :, 1], color="k")

    # DAE with STFT
    ax7 = fig_n.add_subplot(5, 2, 10, sharex=ax1, sharey=ax1)
    ax7.pcolormesh(t_dae_stft, freqs_stft, np.abs(transforms_stft[i, :, :, 2]), shading="auto")
    ax7.set_ylim(0, freq_max)
    ax7.title.set_text("DAE STFT")
    if len(starttimes) == len(signals):
        ax7.set_xlabel("Time since {:04d}-{:02d}-{:02d} 00:00:00".format(obspy.UTCDateTime(str(starttimes[i])).year,
                                                                         obspy.UTCDateTime(str(starttimes[i])).month,
                                                                         obspy.UTCDateTime(str(starttimes[i])).day))
    else:
        ax7.set_xlabel("Time (s)")

    ax8 = fig_n.add_subplot(5, 2, 9, sharex=ax1, sharey=ax2)
    ax8.plot(t_dae_stft_waveform, recovered_stft[i, :, 1], color="k")
    if len(starttimes) == len(signals):
        ax8.set_xlabel("Time since {:04d}-{:02d}-{:02d} 00:00:00".format(obspy.UTCDateTime(str(starttimes[i])).year,
                                                                         obspy.UTCDateTime(str(starttimes[i])).month,
                                                                         obspy.UTCDateTime(str(starttimes[i])).day))
    else:
        ax8.set_xlabel("Time (s)")
    ax8.set_xlim(min(t_signal), max(t_signal))
    plt.tight_layout()

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax3a, ax4a]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # # if snr_cwt - 6 >= snr[i]:
    # name = names[i].split("/")[-1][:-4]
    # filename = "/rscratch/minos14/janis/cwt_denoiser_test_data/figures/GRC4/all_events/{}.png".format(name)
    # fig.savefig(filename)
    #     # if snr_cwt > snr_stft:
    #     #     filename = "/rscratch/minos14/janis/cwt_denoiser_test_data/figures/GRC4/cwt/{}.png".format(name)
    #     #     fig.savefig(filename)
    #
    # plt.close()
    plt.show()
