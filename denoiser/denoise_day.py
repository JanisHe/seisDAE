import numpy as np
import obspy
import copy

from prediction import predict
from utils import load_obj


def denoising_trace(trace, model_filename, config_filename, overlap=0.5, chunksize=None, **kwargs):

    # Load config file
    config = load_obj(config_filename)

    if trace.stats.delta != config["dt"]:
        msg = "Sampling rates of trace {} and denosing model are not equal".format(str(trace))
        raise ValueError(msg)

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
        starttime_list.append(trace.stats.starttime + start * trace.stats.delta)
        data = trace.data[start:]
        data_list.append(np.concatenate((data, np.zeros(config["ts_length"] - len(data)))))

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
