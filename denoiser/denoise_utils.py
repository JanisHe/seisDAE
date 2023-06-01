import os
import numpy as np
import obspy
import copy
import asyncio
import joblib
import pandas as pd
from pathlib import Path

from prediction import predict
from utils import load_obj, is_nan


async def merge_traces(stream: obspy.Stream, header: dict):
    """
    Merging traces in one obspy Stream. Note, altough obspy has a merging routine with good tests, this function is
    faster for the required method.
    In older versions "st_denoised.merge(method=1, fill_value="interpolate")" was used to merge traces of one stream,
    but this line was heavily time consuming.

    :param stream: obspy Stream object that contains all traces that will be merged
    :param header: Dictionary that contains information for the stats of the merged stream. For more information
                   about the stats have a look on the stats of an obspy trace.

    :returns: One obspy stream object instead of several single overlapping traces.
    """

    # Finding length for the resulting array
    array_len = 0
    for trace in stream:
        array_len += trace.stats.npts

    # Allocating emtpy array and putting values into array
    # TODO: Merge traces with arithmetic mean and a small overlap
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


def read_seismic_data(date: obspy.UTCDateTime, sds_dir: str, network: str, station: str, location="*",
                      station_code="EH", channels="ZNE", data_type="D", overlap=False, **kwargs):
    """
    Reads seismic data from SDS structure for a certain date.
    https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html (2022-07-11)

    :param date: Obspy UTCDateTime object to read seismic data from SDS structure
    :param sds_dir: Pathname for SDS directory
    :param network: Name of seismic network
    :param station: Name of seismic station
    :param location: Name of location, default value is *
    :param station_code: Name of the station code, e.g. EH, HH or BH
    :param channels: Name of channels, e.g. ZNE or Z12
    :param data_type: 1 characters indicating the data type, recommended types are:
                    'D' - Waveform data
                    'E' - Detection data
                    'L' - Log data
                    'T' - Timing data
                    'C' - Calibration data
                    'R' - Response data
                    'O' - Opaque data
                    Default is "D". Empty string is also allowed.

    :returns: Obspy stream object that contains all traces. In case of gaps, traces of same IDs are merged and
              gaps are filled by zeros.
    """

    # Add . to data_type if not set in input
    if "." not in data_type:
        data_type = ".{}".format(data_type)

    # Read data
    stream = obspy.Stream()
    for channel in channels:
        stream += obspy.read(os.path.join(sds_dir, "{:04d}".format(date.year), network, station,
                                          f"{station_code}{channel}{data_type}", "*{}*{:03d}".format(location,
                                                                                                    date.julday)),
                             **kwargs)

        # Try to read data from the day before and the next day to add some overlap for denoising
        if overlap is True:
            # Day before
            try:
                day_before = date - 86400
                stime = obspy.UTCDateTime(f"{day_before.date.isoformat()} 23:50")
                stream += obspy.read(os.path.join(sds_dir, "{:04d}".format(day_before.year), network, station,
                                                  f"{station_code}{channel}{data_type}",
                                                  "*{}*{:03d}".format(location, day_before.julday)),
                                     starttime=stime)
            except Exception:
                pass

            # Day after
            try:
                day_after = date + 86400
                etime = obspy.UTCDateTime(f"{day_after.date.isoformat()} 00:10")
                stream += obspy.read(os.path.join(sds_dir, "{:04d}".format(day_after.year), network, station,
                                                  f"{station_code}{channel}{data_type}",
                                                  "*{}*{:03d}".format(location, day_after.julday)),
                                     endtime=etime)
            except Exception:
                pass

    # Merge original stream
    if len(stream) > len(channels):
        for trace in stream:
            trace.data = trace.data - np.mean(trace.data)
        stream.merge(fill_value=0)

    # Sort traces in stream
    stream.sort(keys=['channel'], reverse=True)

    return stream


def denoising_trace(trace, model_filename=None, config_filename=None, overlap=0.8, chunksize=None, verbose=True,
                    **kwargs):
    """
    Denoising of an obspy Trace object using a trained Denoising Autoencoder.

    :param trace: obspy Trace
    :param model_filename: filename of the trained denoising model
    :param config_filename: filename of the config file for the denoising model
    :param overlap: overlap between neighbouring elements in trace [0, 1[
    :param chunksize: int, for denosing of large traces, a trace is splitted into parts of chunksize, otherwise
                      the data might not fit into memory. In particular it is necessary when CWT is used.

    :returns: denoised trace, noisy trace
    """
    # Check value for overlap
    if overlap < 0 or overlap >= 1:
        msg = f"Value overlap is not in range [0, 1[. You set it to {overlap}."
        raise ValueError(msg)

    # Read default model and config file
    if not model_filename or not config_filename:
        model_filename = Path(__file__).parent.parent / "Models" / "gr_mixed_stft.h5"
        config_filename = Path(__file__).parent.parent / "config" / "gr_mixed_stft.config"

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
        for j in range(chunks):
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
    if verbose is True:
        print(f"Successfully denoised {trace.id} between {trace.stats.starttime} and {trace.stats.endtime}")

    return st_denoised[0], st_noise[0]


def denoising_stream(stream, model_filename=None, config_filename=None, overlap=0.8, chunksize=None, parallel=False,
                     verbose=True, **kwargs):
    # TODO: Add pretrained model as model_filename and config_filename (Problem: absolute vs. realtive path)
    """
    Denoises an obspy stream and returns the recovered signal and noise as two separate streams.
    Note, the parameters not mentioned in the description are given in denoising_trace.

    :param stream: obspy stream object
    :param model_filename: Filename of a trained autoencoder
    :param config_filename: Filename of the config file that belongs to the autoencoder
    :param overlap: overlap between neighbouring segments. Default is 0.8
    :param chunksize: If the model has many parameters, e.g. when CWT is used, chunksize splits all 60 s windows
                      into small chunks to reduce the memory. A value of 600 - 800 is recommended. Default is None.
    :param parallel: bool, default is False
                     If True, denoising is done in parallel otherwise one a single CPU
    :returns: stream of recovered signal and noise
    """

    # Check whether input stream is not empty
    if len(stream) == 0:
        msg = "The input stream does not contain any data.\n{}".format(str(stream))
        raise ValueError(msg)

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
                                                   config_filename=config_filename, verbose=verbose,
                                                   overlap=overlap, chunksize=chunksize, **kwargs) for trace in stream)

        # Sort streams from out in stream for recovered signal and noise
        for traces in out:
            st_rec_signal += traces[0]
            st_rec_noise += traces[1]
    else:
        # Loop over each trace in stream and start denoising
        for trace in stream:
            tr_signal, tr_noise = denoising_trace(trace=trace, model_filename=model_filename,
                                                   config_filename=config_filename, verbose=verbose,
                                                   overlap=overlap, chunksize=chunksize, **kwargs)
            st_rec_signal += tr_signal
            st_rec_noise += tr_noise

    return st_rec_signal, st_rec_noise


def denoise(date, model_filename, config_filename, channels, pathname_data, network, station_name,
            station_code, pathname_denoised, station_code_denoised, calib=1.0, noise=False, data_type="", **kwargs):
    """
    Function reads an obspy stream of a certain data and denoises all traces of the stream. Afterwards, the denoised
    stream is saved in pathname_denoised in SDS format, i.e. pathname_data/year/network/station/channel/stream.doy

    :param date: Date to read the input stream of type obspy.UTCDateTime
    :param model_filename: Filename of a trained autoencoder
    :param config_filename: Filename of the config file that belongs to the autoencoder
    :param channels: Names of channels, e.g. ZNE
    :param pathname_data: Full pathname of the raw data
    :param network: Name of the network
    :param station_name: Name of the station
    :param station_code: Station code of the station, e.g. EH or HH
    :param pathname_denoised: Full pathname for the denoised data
    :param station_code_denoised: Station code for the denoised data, e.g. EH or SX
    :param calib: Calibration factor for the denoised data. The denoised data are multiplied with the calib factor to
                  avoid last bits in the denoised data. Default is 1.0
    :param noise: If True the recovered noise the saved, otherwise the recovered signal is saved. Default is False
    :param data_type: Data type of the input data, e.g. D. For more information see Seed manual. Default is an empty
                      string.

    :returns: None
    """
    # Read data for input date
    st_orig = read_seismic_data(date=date, sds_dir=pathname_data, network=network, station=station_name,
                                station_code=station_code, channels=channels, data_type=data_type,
                                overlap=True)

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

    # Trim streams in case of overlapping segments from the day before to same start- and endtime as original data
    # TODO: headonly=True does not always work (e.g. ZB.C22 data). Test whether st_orig_header is empty otherwise read full data
    st_orig_header = read_seismic_data(date=date, sds_dir=pathname_data, network=network, station=station_name,
                                       station_code=station_code, channels=channels, data_type=data_type,
                                       overlap=False, headonly=True)
    for trace_denoised, trace_headonly in zip(st_denoised, st_orig_header):
        if trace_denoised.stats.starttime != trace_headonly.stats.starttime:
            trace_denoised.trim(starttime=trace_headonly.stats.starttime)
        if trace_denoised.stats.endtime != trace_headonly.stats.endtime:
            trace_denoised.trim(endtime=trace_headonly.stats.endtime)

    # Write each denoised trace into single mseed
    for i, denoised in enumerate(st_denoised):
        # Create julday and year
        # Add a few seconds on starttime as for some samples the trace starts at the day before, i.e. 23:59:59.99999
        tmp_date = denoised.stats.starttime + 30  # Add 30 seconds on starttime

        # Create directories if they do not exist
        full_pathname = os.path.join(pathname_denoised, "{:04d}".format(date.year), network, station_name,
                                     f"{denoised.stats.channel}.{data_type}")
        if not os.path.exists(full_pathname):
            os.makedirs(full_pathname)

        # Create filename for the denoised data
        filename = os.path.join(full_pathname,
                                "{}.{}.{}.{}.D.{:04d}.{:03d}".format(denoised.stats.network,
                                                                     denoised.stats.station,
                                                                     denoised.stats.location,
                                                                     denoised.stats.channel,
                                                                     tmp_date.year,
                                                                     tmp_date.julday
                                                                     )
                                )

        # Write full stream
        print("Writing data to {}".format(filename))
        denoised.write(filename=filename,
                       format="MSEED", encoding="STEIM2", byteorder=">", reclen=reclen[i])


def check_endtime(stream1: obspy.Stream, stream2: obspy.Stream, channels="ZNE"):
    """
    Checks whether all traces in two different obspy streams have the same endtime or not.
    Returns False if endtimes are not equal, otherwise True.

    :param stream1: obspy stream
    :param stream2: obspy stream
    :param channels: Channel names of both streams.

    :returns: bool
    """

    if len(stream1) == len(stream2):
        same_endtime = True
        for channel in channels:
            # Note, added some time if a difference exists due do decimation in DAE
            if stream2.select(component=channel)[0].stats.endtime < \
                    stream1.select(component=channel)[0].stats.endtime - stream2[0].stats.delta * 5:
                same_endtime = False
    else:
        same_endtime = False

    return same_endtime


def __auto_denoiser(date: obspy.UTCDateTime, model_filename: str, config_filename: str,
                    sds_dir_noisy: str, sds_dir_denoised: str, network: str, station: str,
                    location="*", station_code_noisy="EH", station_code_denoised="EX", channels="ZNE",
                    data_type="D", calib=1.0, verbose=True, **kwargs):
    """
    Denoises an obspy stream for a given date and writes the denoised traces for each component to an SDS path.

    :param date: obspy UTCDateTime object. Date for which data are read
    :param model_filename: Full pathname of the previously trained denoising autoencoder
    :param config_filename: Full pathname of the config file for the denoising autoencoder
    :param sds_dir_noisy: Base directory of the SDS for reading data
    :param sds_dir_denoised: Base directory of the SDS to write denoised data
    :param network: Name of the network
    :param station: Name of the station
    :param location: Name of location, default value is *
    :param station_code_noisy: Station code of the noisy data, e.g. EH, HH or BH
    :param station_code_denoised: Station code of the denoised data, e.g. EX, HX or SX
    :param channels: Channels of the input data, e.g. ZNE or Z12
    :param data_type: 1 characters indicating the data type, recommended types are:
                    'D' - Waveform data
                    'E' - Detection data
                    'L' - Log data
                    'T' - Timing data
                    'C' - Calibration data
                    'R' - Response data
                    'O' - Opaque data
                    Default is "D". Empty string is also allowed.
    :param calib: Denoised data are mutiplied by this factor in order to avoid last bits in the denoised data.
                  Note, when analysing the data afterwards, the calib factor is not saved in the written trace.
                  Default is 1.0
    :param kwargs: Keywords arguments for the denoiser

    :returns: 1. The full denoised stream for the given date
              2. When some data at the given date are already denoised a trimmed stream that only contains the new
                 denoised data is also returned. This is sometimes necessary, when data are analysed with
                 Seiscomp afterwards and Seiscomps saves the data automatically.
              3. reclen for each trace. Is necessary to save the data in STEIM2.

              If no new data are found, three Nones are returned by this function.

    """

    # Read data for noisy and denoised stream
    # TODO: Set overlap to True in read_seismic_data (merging fails???)
    try:
        noisy_stream = read_seismic_data(date=date, sds_dir=sds_dir_noisy, network=network, station=station,
                                         station_code=station_code_noisy,
                                         channels=channels, data_type=data_type, location=location)
    except Exception as e:
        print(e)
        return None, None, None

    try:
        denoised_stream = read_seismic_data(date=date, sds_dir=sds_dir_denoised, network=network, station=station,
                                            station_code=station_code_denoised, channels=channels,
                                            data_type=data_type, location=location)
    except Exception:
        denoised_stream = obspy.Stream()

    # Check for same endtime
    same_endtime = check_endtime(noisy_stream, denoised_stream, channels)

    # Denoise data in two different ways:
    # 1. If a denoised stream already exists, then the new data are denoised and added to existing data
    # 2. If no denoised stream exists, a new stream is created and written afterwards
    endtime_denoised = None
    if len(denoised_stream) == len(noisy_stream):
        endtime_denoised = []
        for i, trace in enumerate(denoised_stream):
            endtime_denoised_helper = trace.stats.endtime
            noisy_stream[i].trim(starttime=endtime_denoised_helper - 300)  # XXX May lead to masked arrays
            endtime_denoised.append(endtime_denoised_helper)

    # Loop over each trace in original stream and apply denoising method
    if same_endtime is False:
        # Add record length from original stream to empty list
        reclen = []
        for trace in noisy_stream:
            reclen.append(trace.stats.mseed['record_length'])

        # Start denoising for each trace
        denoised, _ = denoising_stream(stream=noisy_stream, model_filename=model_filename,
                                       config_filename=config_filename, verbose=verbose,
                                       overlap=0.8, chunksize=600, parallel=True, **kwargs)

        for trace in denoised:
            trace.stats.channel = "{}{}".format(station_code_denoised, trace.stats.channel[2])
            # Check for nan in denoised array an replace by zeros
            np.nan_to_num(trace.data, copy=False, nan=0.0)
            # Multiply data in trace by calib to avoid lat bit
            if float(calib) != 1.0:
                trace.data = trace.data * calib
            # Change dtype for STEIM2 encoding
            trace.data = trace.data.astype("int32")
            denoised_stream += trace

        # Merge all traces of same id in denoised stream
        denoised_stream.merge(method=1, fill_value=0)
        denoised_stream.sort(keys=['channel'], reverse=True)

        # Write each denoised trace into single mseed
        for i, denoised in enumerate(denoised_stream):
            # Make directories if they do not exist
            full_pathname = os.path.join(sds_dir_denoised, "{:04d}".format(date.year), network, station,
                                         f"{denoised.stats.channel}{data_type}")
            if not os.path.exists(full_pathname):
                os.makedirs(full_pathname)

            filename = os.path.join("{}{:04d}".format(sds_dir_denoised, date.year), network, station,
                                    f"{denoised.stats.channel}{data_type}", "{}.{}.{}.{}.D.{:04d}.{:03d}".
                                    format(denoised.stats.network,
                                    denoised.stats.station,
                                    denoised.stats.location,
                                    denoised.stats.channel,
                                    denoised.stats.starttime.year,
                                    denoised.stats.starttime.julday)
                                    )

            # Write full stream
            denoised.write(filename=filename,
                           format="MSEED", encoding="STEIM2", byteorder=">", reclen=reclen[i])

        # Return denoised stream and a copy that only contains the new denoised data
        # Both returned values are obspy streams
        denoised_stream_cp = denoised_stream.copy()
        if endtime_denoised:
            for i, trace in enumerate(denoised_stream_cp):
                trace.trim(starttime=endtime_denoised[i] + denoised.stats.delta)

        return denoised_stream, denoised_stream_cp, reclen
    else:
        return None, None, None


def read_csv(filename: str, date=obspy.UTCDateTime(), **kwargs):
    """
    Function reads a csv file for the auto denoiser and returns a dictionary for each station that is given in
    the input csv file. Keyword arguments are for pandas.read_csv function.
    """

    # Read csv file
    df_csv = pd.read_csv(filename, **kwargs)

    # Create a dictionary for each station in df_csv / csv_file
    df_dict = {}
    for i, station in enumerate(df_csv["station"]):
        network = df_csv["network"][i]

        # Add . to data_type if not set in input
        if df_csv['type'][i] != ".":
            data_type = ".{}".format(df_csv['type'][i])
        else:
            data_type = df_csv['type'][i]

        # Add location to dataframe if it is available in csv file, otherwise use *
        try:
            location = df_csv['location'][i]
            if is_nan(location):
                location = "*"
            if isinstance(location, float):                               # Convert location to two digit string
                location = "{:02d}".format(int(location))
        except KeyError:
            location = "*"

        # Update dict
        df_dict.update({"{}.{}".format(network, station): dict(date=date,
                                                               model_filename=df_csv['dae_model'][i],
                                                               config_filename=df_csv['config'][i],
                                                               sds_dir_noisy=df_csv['sdsdir'][i],
                                                               sds_dir_denoised=df_csv['sds_out'][i],
                                                               network=df_csv['network'][i],
                                                               station=df_csv['station'][i],
                                                               location=location,
                                                               station_code_noisy=df_csv['channel_code'][i],
                                                               station_code_denoised=df_csv['channel_code_denoised'][i],
                                                               channels=df_csv['channel_direction'][i],
                                                               data_type=data_type,
                                                               calib=float(df_csv['calib'][i])
                                                               )
                        }
                       )

    return df_dict


def auto_denoiser(csv_file: str, date: obspy.UTCDateTime, n_cores=1):
    """
    Starts automatic denoising by reading parameters for each station from a csv file.
    Which data are read is controlled by the given date as this is not defined in the csv file.
    This function can run in parallel by joblib but it is not recommended as it might lead to an overload.

    :param csv_file: Full filename of the input csv file
    :param date: obspy UTCDateTime for denoising data
    :param n_cores: Number of cores to run denoising in parallel. Be careful when denoising several stations in
                    parallel, as method denosing_stream is already parallelised to speed up computation time.
                    Default is 1 core. Higher numbers might result in an overload.

    :returns: All traces and results that are returned by the function _auto_denoiser
    """

    # Create a dictionary for each station in df_csv / csv_file
    auto_denoiser_dict = read_csv(csv_file, date, delimiter=",", comment="#")

    # Start auto denoiser for each station in auto_denoiser_dict
    pool = joblib.Parallel(n_jobs=n_cores, backend="multiprocessing", prefer="processes")
    out = pool(joblib.delayed(__auto_denoiser)(**auto_denoiser_dict[item]) for item in auto_denoiser_dict)

    return out
