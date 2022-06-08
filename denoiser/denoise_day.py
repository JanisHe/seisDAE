import os
import numpy as np
import obspy
import copy
import asyncio
from tqdm import tqdm
import joblib

from prediction import predict
from utils import load_obj


async def merge_traces(stream: obspy.Stream, header: dict):
    """
    Merging traces in one obspy Stream. Note, altough obspy has a merging routine with good tests, this function is
    faster for the required method.
    In older versions "st_denoised.merge(method=1, fill_value="interpolate")" was used to merge traces of one stream,
    but this line was heavily time consuming.

    :param stream:
    :param header:

    :returns
    """

    # Finding length for the resulting array
    array_len = 0
    for trace in stream:
        array_len += trace.stats.npts

    # Allocating emtpy array and putting values into array
    data = np.zeros(array_len)
    start = 0
    for trace in stream:
        data[start:start + trace.stats.npts] = trace.data
        start += trace.stats.npts

    # Create obspy stream
    stream_out = obspy.Stream()
    trace_out = obspy.Trace(data=data, header=header)
    stream_out += trace_out

    return stream_out


def denoising_trace(trace, model_filename, config_filename, overlap=0.8, chunksize=None, **kwargs):
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
                tr_den.trim(starttime=st_denoised[-1].stats.endtime + dt)
                tr_noi.trim(starttime=st_noise[-1].stats.endtime + dt)
            else:
                tr_den.trim(starttime=st_denoised[-1].stats.endtime + dt,
                            endtime=tr_den.stats.endtime - tr_den.stats.npts * tr_den.stats.delta * overlap / 2)
                tr_noi.trim(starttime=st_noise[-1].stats.endtime + dt,
                            endtime=tr_noi.stats.endtime - tr_noi.stats.npts * tr_noi.stats.delta * overlap / 2)

        st_denoised += tr_den
        st_noise += tr_noi

    # Merge both streams and return streams for recovered signal and noise
    # Both merges are done in parallel with asyncIO
    merge_loop = asyncio.get_event_loop()
    st_denoised = merge_loop.run_until_complete(merge_traces(st_denoised, header=dict(delta=dt,
                                                                                      starttime=starttime_list[0],
                                                                                      station=trace.stats.station,
                                                                                      network=trace.stats.network,
                                                                                      location=trace.stats.location,
                                                                                      channel=trace.stats.channel)))

    st_noise = merge_loop.run_until_complete(merge_traces(st_noise, header=dict(delta=dt,
                                                                                starttime=starttime_list[0],
                                                                                station=trace.stats.station,
                                                                                network=trace.stats.network,
                                                                                location=trace.stats.location,
                                                                                channel=trace.stats.channel)))

    return st_denoised[0], st_noise[0]


def denoising_stream(stream, model_filename, config_filename, overlap=0.8, chunksize=None, parallel=False, **kwargs):
    """
    Denoises an obspy stream and returns the recovered signal and noise as two separate streams.
    Note, the parameters not mentioned in the description are given in denoising_trace.

    :param parallel: bool, default is False
                     If True, denoising is done in parallel otherwise one a single CPU
    :returns: stream of recovered signal and noise
    """

    st_rec_signal = obspy.Stream()
    st_rec_noise = obspy.Stream()

    # Finding maximum number of CPUs
    if parallel is True:
        if len(stream) <= int(os.cpu_count() / 2):
            n_jobs = len(stream)
        else:
            n_jobs = int(os.cpu_count() / 2)

        # Run denoising for each trace in parallel
        pool = joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
        out = pool(joblib.delayed(denoising_trace)(trace=trace, model_filename=model_filename,
                                                   config_filename=config_filename,
                                                   overlap=overlap, chunksize=chunksize, **kwargs) for trace in stream)

        # Sort streams from out in stream for recovered signal and noise
        for traces in out:
            st_rec_signal += traces[0]
            st_rec_noise += traces[1]
    else:
        # Loop over each trace in stream and start denoising
        for trace in stream:
            tr_signal, tr_noise = denoising_trace(trace=trace, model_filename=model_filename,
                                                   config_filename=config_filename,
                                                   overlap=overlap, chunksize=chunksize, **kwargs)
            st_rec_signal += tr_signal
            st_rec_noise += tr_noise

    return st_rec_signal, st_rec_noise


def denoise(date, model_filename, config_filename, channels, pathname_data, network, station_name,
            station_code, pathname_denoised, station_code_denoised, calib=1.0, noise=False, data_type="", **kwargs):
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
        denoised, noisy = denoising_trace(trace=trace, model_filename=model_filename, config_filename=config_filename,
                                          overlap=0.8, chunksize=600, **kwargs)
        if noise is False:
            denoised.stats.channel = "{}{}".format(station_code_denoised, trace.stats.channel[2])
            # Check for nan in denoised array an replace by zeros
            np.nan_to_num(denoised.data, copy=False, nan=0.0)
            # Change dtype for STEIM2 encoding
            denoised.data = denoised.data * calib
            denoised.data = denoised.data.astype("int32")
            reclen.append(trace.stats.mseed['record_length'])
            st_denoised += denoised
        else:
            noisy.stats.channel = "{}{}".format(station_code_denoised, trace.stats.channel[2])
            # Check for nan in denoised array an replace by zeros
            np.nan_to_num(noisy.data, copy=False, nan=0.0)
            # Change dtype for STEIM2 encoding
            noisy.data = noisy.data * calib
            noisy.data = noisy.data.astype("int32")
            reclen.append(trace.stats.mseed['record_length'])
            st_denoised += noisy

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

        # Write full stream
        print("Writing data to {}".format(filename))
        denoised.write(filename=filename,
                       format="MSEED", encoding="STEIM2", byteorder=">", reclen=reclen[i])


if __name__ == "__main__":
    from obspy import UTCDateTime, read, Stream
    from obspy.clients.filesystem.sds import Client as LocalClient
    import matplotlib.pyplot as plt
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    starttime = obspy.UTCDateTime("2020-03-04 08:11")
    endtime = starttime + 4*3600

    client = LocalClient("/data/denoise/data/raw_data/SDS/")
    st = client.get_waveforms(network="BW", station="ROTZ", location="*", channel="HH?", starttime=starttime,
                              endtime=endtime)

    d_stime = time.time()
    # XXX Function that takes stream and denoises each stream in parallel instead
    st_denoised, st_noise = denoising_stream(st, model_filename="/home/janis/CODE/cwt_denoiser/Models/BW_ROTZ_stft.h5",
                                             config_filename="/home/janis/CODE/cwt_denoiser/config/BW_ROTZ_stft.config",
                                             overlap=0.8, parallel=False, ckpt_model=False)
    d_etime = time.time()
    print("needed {} s for denoising".format(d_etime - d_stime))

    fig = plt.figure()
    # Plot original stream
    c = 1
    for i, tr in enumerate(st):
        if i == 0:
            ax = fig.add_subplot(3, 2, i + c)
        else:
            ax = fig.add_subplot(3, 2, i + c, sharex=ax)
        ax.plot(tr.data, color="k")
        c += 1

    # Plot denoised stream
    c = 2
    for i, tr in enumerate(st_denoised):
        ax = fig.add_subplot(3, 2, i + c, sharex=ax)
        ax.plot(tr.data, color="k")
        c += 1

    plt.show()
