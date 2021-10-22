import os
import numpy as np
import obspy
import copy
from tqdm import tqdm

from prediction import predict
from utils import load_obj


def denoising_trace(trace, model_filename, config_filename, overlap=0.5, chunksize=None, **kwargs):
    """
    Denoising of an obspy Trace object using a trained Denoising Autoencoder.

    :param trace: obspy Trace
    :param model_filename: filename of the trained denoising model
    :param config_filename: filename of the config file for the denoising model
    :param overlap: overlap between neighbouring elements in trace [0, 1]
    :param chunksize: int, for denosing of large traces, a trace is splitted into parts of chunksize, otherwise
                      the data might not fit into memory.

    :returns: denoised trace, noisy trace
    """

    # Load config file
    config = load_obj(config_filename)

    # if trace.stats.delta != config["dt"]:
    #     msg = "Sampling rates of trace {} and denosing model are not equal".format(str(trace))
    #     raise ValueError(msg)

    # Resample trace if sampling rate in config file and sampling rate of trace are not equal
    if trace.stats.delta != config['dt']:
        trace.resample(sampling_rate=1 / config['dt'])

    # Loop over each window
    data_list = []
    starttime_list = []
    start = 0
    end = config["ts_length"]
    while end <= trace.stats.npts:
        data_list.append(trace.data[start:end])
        starttime_list.append(trace.stats.starttime + start * trace.stats.delta)
        start += int(config["ts_length"] * (1 - overlap))
        end = start + config["ts_length"]

    if end + 1 > trace.stats.npts:
        start = trace.stats.npts - config['ts_length']
        starttime_list.append(trace.stats.starttime + start * trace.stats.delta)
        data_list.append(trace.data[start:])
        #data_list.append(np.concatenate((data, np.zeros(config["ts_length"] - len(data)))))

    # Denoise data by prediction
    if chunksize is not None and chunksize >= 1:
        chunks = int(np.ceil(len(data_list) / chunksize))
        for j in tqdm(range(chunks)):
            d = data_list[int(j*chunksize):int((j+1)*chunksize)]
            r, _, _ = predict(model_filename, config_filename, d, **kwargs)
            if j == 0:
                recovered = copy.copy(r)
            else:
                recovered = np.concatenate((recovered, r))
    else:
        recovered, _, _ = predict(model_filename, config_filename, data_list, **kwargs)

    # Recover denoised and noise stream
    st_denoised = obspy.Stream()
    st_noise = obspy.Stream()

    if config["decimation_factor"] is not None:
        dt = trace.stats.delta * config["decimation_factor"]
    else:
        dt = trace.stats.delta

    for i in range(recovered.shape[0]):
        tr_den = obspy.Trace(data=recovered[i, :, 0], header=dict(delta=dt, starttime=starttime_list[i],
                                                                  station=trace.stats.station,
                                                                  network=trace.stats.network,
                                                                  location=trace.stats.location,
                                                                  channel=trace.stats.channel)
                             )
        tr_noi = obspy.Trace(data=recovered[i, :, 1], header=dict(delta=dt, starttime=starttime_list[i],
                                                                  station=trace.stats.station,
                                                                  network=trace.stats.network,
                                                                  location=trace.stats.location,
                                                                  channel=trace.stats.channel)
                             )

        # Trim traces if overlap is not 0 to merge traces later
        if overlap > 0:
            if i == 0:
                tr_den.trim(endtime=tr_den.stats.endtime - tr_den.stats.npts * tr_den.stats.delta * overlap / 2)
                tr_noi.trim(endtime=tr_noi.stats.endtime - tr_noi.stats.npts * tr_noi.stats.delta * overlap / 2)
            elif i == recovered.shape[0] - 1:
                tr_den.trim(starttime=tr_den.stats.starttime + tr_den.stats.npts * tr_den.stats.delta * overlap / 2)
                tr_noi.trim(starttime=tr_noi.stats.starttime + tr_noi.stats.npts * tr_noi.stats.delta * overlap / 2)
            else:
                tr_den.trim(starttime=tr_den.stats.starttime + tr_den.stats.npts * tr_den.stats.delta * overlap / 2,
                            endtime=tr_den.stats.endtime - tr_den.stats.npts * tr_den.stats.delta * overlap / 2)
                tr_noi.trim(starttime=tr_noi.stats.starttime + tr_noi.stats.npts * tr_noi.stats.delta * overlap / 2,
                            endtime=tr_noi.stats.endtime - tr_noi.stats.npts * tr_noi.stats.delta * overlap / 2)

        st_denoised += tr_den
        st_noise += tr_noi

    # Merge both streams and return traces
    st_denoised.merge(fill_value="interpolate")
    st_noise.merge(fill_value="interpolate")

    # Trim streams on start- and endtime
    st_denoised.trim(starttime=trace.stats.starttime, endtime=trace.stats.endtime)
    st_noise.trim(starttime=trace.stats.starttime, endtime=trace.stats.endtime)

    return st_denoised[0], st_noise[0]


def denoise(date, model_filename, config_filename, channels, pathname_data, network, station_name,
            station_code, pathname_denoised, station_code_denoised, data_type="", **kwargs):
    """
    """
    # Read data for today
    st_orig = obspy.Stream()
    for channel in channels:
        st_orig += obspy.read("{}/{:04d}/{}/{}/{}{}{}/*{:03d}".format(pathname_data, date.year, network, station_name,
                                                                      station_code, channel, data_type,
                                                                      date.julday))
    # Merge original stream
    if len(st_orig) > len(channels):
        for trace in st_orig:
            trace.data = trace.data - np.mean(trace.data)
        st_orig.merge(fill_value=0)


    # Sort streams in same order, thus later loops have the same indices for st_orig and st_denoised
    st_orig.sort(keys=['channel'], reverse=True)

    # Denoise original stream object
    # Loop over each trace in original stream and apply denoising method
    st_denoised = obspy.Stream()
    reclen = []
    for trace in st_orig:
        denoised, _ = denoising_trace(trace=trace, model_filename=model_filename, config_filename=config_filename,
                                      overlap=0.6, chunksize=600, **kwargs)
        denoised.stats.channel = "{}{}".format(station_code_denoised, trace.stats.channel[2])
        # Check for nan in denoised array an replace by zeros
        np.nan_to_num(denoised.data, copy=False, nan=0.0)
        # Change dtype for STEIM2 encoding
        denoised.data = denoised.data.astype("int32")
        reclen.append(trace.stats.mseed['record_length'])
        st_denoised += denoised
    st_denoised.merge(method=1, fill_value=0)
    st_denoised.sort(keys=['channel'], reverse=True)

    # Write each denoised trace into single mseed
    for i, denoised in enumerate(st_denoised):
        # Make directories if they do not exist
        full_pathname = "{}{:04d}/{}/{}/{}{}".format(pathname_denoised, date.year, network, station_name,
                                                     denoised.stats.channel, data_type)
        if not os.path.exists(full_pathname):
            os.makedirs(full_pathname)

        filename = "{}{:04d}/{}/{}/{}{}/{}.{}.{}.{}.D.{:04d}.{:03d}".format(pathname_denoised, date.year, network,
                                                                            station_name, denoised.stats.channel,
                                                                            data_type,
                                                                            denoised.stats.network,
                                                                            denoised.stats.station,
                                                                            denoised.stats.location,
                                                                            denoised.stats.channel,
                                                                            denoised.stats.starttime.year,
                                                                            denoised.stats.starttime.julday
                                                                            )

        #denoised.trim(endtime=denoised.stats.endtime - 60)  # XXX
        # Write full stream
        denoised.write(filename=filename,
                       format="MSEED", encoding="STEIM2", byteorder=">", reclen=reclen[i])
