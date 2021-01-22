import pickle

def save_obj(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=0)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



if __name__ == "__main__":
    d = dict(ts_length=6001, use_bias=False, activation=None, drop_rate=0.0, channels=2, optimizer="adam",
              loss='mean_squared_error', filter_root=8, depth=6, fully_connected=False, max_pooling=False,
              dt=0.01, decimation_factor=None, cwt=False, kwargs=dict(nfft=61, noverlap=16, nperseg=31),
             kernel_size=(3, 3), strides=(2, 2))
    save_obj(d, "./config/2021-01-20_stft.config")

