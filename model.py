"""
Test for Denoising AuotEncoder. Input channels are absolute value and phase of STFT
"""

import os
import glob
import copy
import random

import obspy
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from scipy.signal import stft, istft
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model as tfmodel
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Dropout, Conv2DTranspose, Cropping2D, \
    MaxPooling2D, UpSampling2D, Dense, Softmax, Flatten, Reshape

from pycwt import pycwt
from utils import save_obj


def cwt_wrapper(x, dt=1.0, yshape=150, **kwargs):

    # Remove mean from x
    x = x - np.mean(x)

    # Frequencies for CWT with numpy logspace
    # XXX Effect of logspace instead of linspace ????
    #freqs = np.logspace(start=np.log10(dt), stop=np.log10(1 / (2 * dt)), num=yshape)
    freqs = np.linspace(dt, 1 / (2 * dt), yshape)

    # Transforming x to TF-doamin
    coeffs, scales, freqs_x, _, _, _ = pycwt.cwt(x, dt=dt, freqs=freqs, **kwargs)

    # Estimate dj as eq (9) & (10) in Torrence & Compo
    dj = 1 / yshape * np.log2(len(x) * dt / np.min(scales))

    return coeffs, scales, dj, freqs


def random_float(low, high):
    return random.random()*(high-low) + low


def cropping_layer(needed_shape, is_shape):
    diff1 = is_shape[0] - needed_shape[0]
    if diff1 % 2 == 0 and diff1 > 0:
        shape1 = (diff1//2, diff1//2)
    elif diff1 % 2 == 1 and diff1 > 0:
        shape1 = (diff1//2, is_shape[0] - needed_shape[0])
    elif diff1 == 0:
        shape1 = (0, 0)

    diff2 = is_shape[1] - needed_shape[1]
    if diff2 % 2 == 0 and diff2 > 0:
        shape2 = (diff2//2, diff2//2)
    elif diff2 % 2 == 1 and diff2 > 0:
        shape2 = (diff1//2, is_shape[1] - needed_shape[1])
    elif diff2 == 0:
        shape2 = (0, 0)

    return shape1, shape2



class Model:

    def __init__(self, ts_length=6001, dt=1.0, validation_split=0.15, optimizer="adam",
                 loss='mean_absolute_error', activation="sigmoid", drop_rate=0.001,
                 use_bias=False, data_augmentation=True, shuffle=True, channels=2, decimation_factor=2, cwt=False,
                 callbacks=None, **kwargs):

        self.dt = dt
        self.dt_orig = dt
        self.ts_length = ts_length
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.channels = channels
        self.decimation_factor = decimation_factor
        self.cwt = cwt
        self.kwargs = kwargs
        self.history = None
        self.callbacks = callbacks
        self.depth = None
        self.kernel_size = None
        self.strides = None
        self.filter_root = None
        self.fully_connected = None
        self.max_pooling = None


        dummy = obspy.Trace(data=np.random.rand(ts_length), header=dict(delta=self.dt))
        if decimation_factor:
            dummy.decimate(factor=decimation_factor)
            # Get new dt after decimation
            self.dt = dummy.stats.delta

        if self.cwt is False:
            _, _, dummystft = stft(dummy.data, fs=1 / self.dt, **kwargs)
            self.shape = dummystft.shape
            # Test whether stft is invertible and results in same length as input length
            t, dummy_x = istft(Zxx=dummystft, fs=1 / self.dt, **kwargs)
            if len(dummy_x) != len(dummy.data):
                msg = "Keywordarguments of STFT and ISTFT do not fit. \nThus, length of inverse STFT is {} which is " \
                      "not equal with length {} if input data.\nThis might lead to an error, when applying the " \
                      "trained model.\nPlease change your keywordarguments.".format(len(dummy_x), len(dummy.data))
                raise ValueError(msg)
        elif self.cwt is True:
            dummy_coeff, _, _, _ = cwt_wrapper(x=dummy.data, dt=self.dt, **kwargs)
            self.shape = (dummy_coeff.shape[0], dummy_coeff.shape[1])

    def build_model(self, filter_root=8, depth=4, kernel_size=(3, 3), fully_connected=False, strides=(2, 2),
                    max_pooling=False, **kwargs):

        self.filter_root = filter_root
        self.depth = depth
        self.kernel_size = kernel_size
        self.fully_connected = fully_connected
        self.strides = strides
        self.max_pooling = max_pooling

        # Pooling vs stride for downsampling:
        # https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
        if max_pooling is True:
            pool_size = copy.copy(strides)
            strides = (1, 1)

        # Define Input layer
        input_layer = Input((self.shape[0], self.shape[1], self.channels))

        # Empty dict to save shape for each layer
        layer_shapes = dict()

        # XXX Dropout just in Decoder???
        # Encoder
        h = Conv2D(filter_root, kernel_size, activation=self.activation, padding='same',
                   use_bias=self.use_bias)(input_layer)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(rate=self.drop_rate)(h)

        for i in range(depth + 1):
            h = Conv2D(int(2**i * filter_root), kernel_size, activation=self.activation, padding='same',
                       use_bias=self.use_bias)(h)
            h = ReLU()(h)
            h = BatchNormalization()(h)
            h = Dropout(rate=self.drop_rate)(h)

            h = Conv2D(int(2**i * filter_root), kernel_size, activation=self.activation, padding='same',
                       use_bias=self.use_bias, strides=strides)(h)

            if max_pooling is True:
                h = MaxPooling2D(pool_size=pool_size, padding="same")(h)

            h = ReLU()(h)
            h = BatchNormalization()(h)
            h = Dropout(rate=self.drop_rate)(h)

            layer_shapes.update({i: (h.shape[1], h.shape[2])})

        # Fully Connected Layer
        if fully_connected is True:
            conv_shape = tuple(h.shape[1:])
            h = Flatten()(h)
            flatten_shape = h.shape[1]
            h = Dense(units=flatten_shape, activation=self.activation, use_bias=self.use_bias)(h)
            h = Dense(units=int(flatten_shape / 10), activation=self.activation, use_bias=self.use_bias)(h)
            h = Dense(units=flatten_shape, activation=self.activation, use_bias=self.use_bias)(h)
            h = Reshape(target_shape=conv_shape)(h)

        # Decoder
        for i in range(depth-1, -1, -1):
            needed_shape = layer_shapes[i]

            h = Conv2DTranspose(int(2**i * filter_root), kernel_size, activation=self.activation, padding="same",
                                use_bias=self.use_bias)(h)
            h = ReLU()(h)
            h = BatchNormalization()(h)
            h = Dropout(rate=self.drop_rate)(h)

            if max_pooling is True:
                h = Conv2D(int(2**i * filter_root), kernel_size, activation=self.activation, padding='same',
                                use_bias=self.use_bias, strides=strides)(h)
                h = UpSampling2D(size=pool_size)(h)
            elif max_pooling is False:
                h = Conv2DTranspose(int(2**i * filter_root), kernel_size, activation=self.activation, padding='same',
                                    use_bias=self.use_bias, strides=strides)(h)

            h = ReLU()(h)
            h = BatchNormalization()(h)
            h = Dropout(rate=self.drop_rate)(h)
            crop = cropping_layer(needed_shape, is_shape=(h.shape[1], h.shape[2]))
            h = Cropping2D(cropping=(crop[0], crop[1]))(h)

        h = Conv2DTranspose(filter_root, kernel_size, activation=self.activation, padding="same",
                            use_bias=self.use_bias)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(rate=self.drop_rate)(h)

        if max_pooling is True:
            h = Conv2D(filter_root, kernel_size, activation=self.activation, padding='same',
                            use_bias=self.use_bias, strides=strides)(h)
            h = UpSampling2D(size=pool_size)(h)
        elif max_pooling is False:
            h = Conv2DTranspose(filter_root, kernel_size, activation=self.activation, padding='same',
                                use_bias=self.use_bias, strides=strides)(h)

        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(rate=self.drop_rate)(h)
        crop = cropping_layer(needed_shape=(input_layer.shape[1], input_layer.shape[2]),
                              is_shape=(h.shape[1], h.shape[2]))
        h = Cropping2D(cropping=(crop[0], crop[1]))(h)

        h = Conv2D(self.channels, (1, 1), activation=self.activation, padding='same',
                   use_bias=self.use_bias)(h)
        h = Softmax()(h)

        # Build model and compile Model
        self.model = tfmodel(input_layer, h)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])


    def summarize(self):
        self.model.summary()

    def save_config(self, pathname="./config", filename=None):
        if filename:
            settings_filename = "{}/{}".format(pathname, filename)
        else:
            if self.cwt is False:
                settings_filename = "{}/{}_stft.config".format(pathname, str(date.today()))
            elif self.cwt is True:
                settings_filename = "{}/{}_cwt.config".format(pathname, str(date.today()))

        # Write all important parameters to config file
        if isinstance(self.optimizer, str) is True:
            optimizer_name = self.optimizer
        else:
            optimizer_name = self.optimizer._name
            
        config_dict = dict(shape=self.shape, ts_length=self.ts_length, dt=self.dt_orig, channels=self.channels,
                           depth=self.depth, filter_root=self.filter_root, kernel_size=self.kernel_size,
                           strides=self.strides, optimizer=optimizer_name, fully_connected=self.fully_connected,
                           use_bias=self.use_bias, loss=self.loss, activation=self.activation,
                           drop_rate=self.drop_rate, decimation_factor=self.decimation_factor,
                           max_pooling=self.max_pooling, cwt=self.cwt, kwargs=self.kwargs,
                           data_augmentation=self.data_augmentation)
        save_obj(dictionary=config_dict, filename=settings_filename)

    def save_model(self, pathname_model="./Models", pathname_config="./config"):
        """
        Save model as .h5 file and write a .txt file with all important settings.
        """
        # Save config file
        self.save_config(pathname=pathname_config)
        # Save fully trained model
        if self.cwt is False:
            self.model.save("{}/{}_stft.h5".format(pathname_model, str(date.today())), overwrite=True)
        elif self.cwt is True:
            self.model.save("{}/{}_cwt.h5".format(pathname_model, str(date.today())), overwrite=True)


    def train_model_generator(self, signal_file, noise_file,
                              epochs=50, batch_size=20, validation_split=0.15, verbose=1,
                              workers=8, use_multiprocessing=True):

        # Save config file in config directory as tmp.config
        self.save_config(pathname="./config", filename="tmp.config")

        # Split value to split data into training and validation datasets
        split = int(len(signal_file) * (1 - validation_split))

        # Shuffle list randomly to get different data for validation
        if self.shuffle is True:
            random.shuffle(signal_file)

        # Generate data for each batch
        generator_train = DataGenerator(signal_list=signal_file[:split], noise_list=noise_file,
                                        batch_size=batch_size,
                                        shape=self.shape, data_augmentation=self.data_augmentation,
                                        cwt=self.cwt, dt=self.dt, ts_length=self.ts_length,
                                        decimation_factor=self.decimation_factor,
                                        **self.kwargs)
        generator_validate = DataGenerator(signal_list=signal_file[split:], noise_list=noise_file,
                                           batch_size=batch_size,
                                           shape=self.shape, data_augmentation=self.data_augmentation,
                                           cwt=self.cwt, dt=self.dt, ts_length=self.ts_length,
                                           decimation_factor=self.decimation_factor,
                                           **self.kwargs)

        self.history = self.model.fit(x=generator_train, epochs=epochs, workers=workers,
                                      use_multiprocessing=use_multiprocessing,
                                      verbose=verbose, validation_data=generator_validate,
                                      callbacks=self.callbacks)

        # Remove temporary config file
        os.remove("./config/tmp.config")

    def plot_history(self, pathname="./figures", plot=True):
        """
        Plot loss vs epochs of training and validation
        """
        # summarize history for accuracy
        fig_acc = plt.figure()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        if plot is True:
            if self.cwt is False:
                plt.savefig("{}/{}_stft_accuracy.png".format(pathname, str(date.today())))
            elif self.cwt is True:
                plt.savefig("{}/{}_cwt_accuracy.png".format(pathname, str(date.today())))

        # summarize history for loss
        fig_loss = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        if plot is True:
            if self.cwt is False:
                plt.savefig("{}/{}_stft_loss.png".format(pathname, str(date.today())))
            elif self.cwt is True:
                plt.savefig("{}/{}_cwt_loss.png".format(pathname, str(date.today())))

        if plot is not True:
            return fig_acc, fig_loss


class DataGenerator(Sequence):

    def __init__(self, signal_list, noise_list, batch_size=20, shape=(90, 6001), channels=2, decimation_factor=2,
                 cwt=False, dt=1.0, ts_length=6001, data_augmentation=True, **kwargs):

        self.shape = shape
        self.batch_size = batch_size
        self.signal_list = signal_list
        self.noise_list = glob.glob(noise_list)
        self.channels = channels
        self.decimation_factor = decimation_factor
        self.cwt = cwt
        self.dt = dt
        self.ts_length = ts_length
        self.data_augmentation = data_augmentation
        self.kwargs = kwargs

        if len(noise_list) == 0:
            msg = "Could not load noise files from {}".format(noise_list)
            raise ValueError(msg)

    def __len__(self):
        return int(np.floor(len(self.signal_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        return self.__data_generation()

    def __data_generation(self):
        X = np.empty(shape=(self.batch_size, *self.shape, self.channels), dtype="float")   # Empty input data
        Y = np.empty(shape=(self.batch_size, *self.shape, self.channels), dtype="float")   # Empty target data

        for i in range(self.batch_size):
            signal_filename = "{}".format(self.signal_list[random.randint(0, len(self.signal_list) - 1)])
            noise_filename = "{}".format(self.noise_list[random.randint(0, len(self.noise_list) - 1)])
            signal = np.load(signal_filename)
            noise = np.load(noise_filename)

            # Read signal and noise from npz files
            signal = signal.f.data[:self.ts_length]
            noise = noise.f.data[:self.ts_length]

            # XXX Leads to Runtime Warnings (Division by zero) in estimation of mapping functions
            # XXX RuntimeWarning: divide by zero encountered in true_divide
            # DATA AUGMENTATION
            # Move signal randomly, hence P-arrival varies its place
            # Add randomly zeros at beginning
            if self.data_augmentation is True:
                add_zeros = np.random.uniform(low=-1, high=1,
                                              size=(random.randint(0, int(self.ts_length / 2)), )) * 1e-15
                signal = np.concatenate((add_zeros, signal))
                # Cut signal to length of ts_length
                signal = signal[:self.ts_length]

            # Remove mean
            noise = noise - np.mean(noise)
            signal = signal - np.mean(signal)

            # Apply highpass filter
            tr_n = obspy.Trace(data=noise, header=dict(delta=self.dt))
            tr_s = obspy.Trace(data=signal, header=dict(delta=self.dt))
            tr_n.filter("highpass", freq=0.5)
            tr_s.filter("highpass", freq=0.5)

            # Decimate by factor decimation_factor
            if self.decimation_factor:
                tr_n.decimate(factor=self.decimation_factor)
                tr_s.decimate(factor=self.decimation_factor)

            # XXX Test some different normalizing
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
            signal = signal / norm
            noise = noise / norm

            # STFT or CWT of noisy signal and signal
            if self.cwt is False:
                _, _, cns = stft(noisy_signal, fs=1 / self.dt, **self.kwargs)
                _, _, cs = stft(signal, fs=1 / self.dt, **self.kwargs)
                _, _, cn = stft(noise, fs=1 / self.dt, **self.kwargs)
            elif self.cwt is True:
                cns, _, _, _ = cwt_wrapper(noisy_signal, dt=self.dt, **self.kwargs)
                cs, _, _, _ = cwt_wrapper(signal, dt=self.dt, **self.kwargs)
                cn, _, _, _ = cwt_wrapper(noise, dt=self.dt, **self.kwargs)

            # Write data to empty np arrays
            # Zhu et al, 2018
            X[i, :, :, 0] = cns.real
            Y[i, :, :, 0] = 1 / (1 + np.abs(cn) / np.abs(cs))

            # X[i, :, :, 0] = np.abs(cns)
            # Y[i, :, :, 0] = np.abs(cs)

            # X[i, :, :, 0] = np.abs(cns)
            # Y[i, :, :, 0] = np.abs(cs) / np.abs(cns)

            Y[i, :, :, 0] = np.nan_to_num(Y[i, :, :, 0])

            if self.channels == 2:
                # Zhu et al, 2018
                X[i, :, :, 1] = cns.imag
                Y[i, :, :, 1] = (np.abs(cn) / np.abs(cs)) / (1 + np.abs(cn) / np.abs(cs))

                # X[i, :, :, 1] = np.arctan2(cns.imag, cns.real)
                # Y[i, :, :, 1] = np.arctan2(cs.imag, cs.real)

                Y[i, :, :, 1] = np.nan_to_num(Y[i, :, :, 1])
            else:
                msg = "Channel number cannot exceed 2."
                raise ValueError(msg)

        return X, Y



if __name__ == "__main__":

    #"/home/geophysik/Schreibtisch/denoiser_data/"

    signal_files = glob.glob("/rscratch/minos14/janis/dae_noise_data/signal/*")[:600]
    noise_files = "/rscratch/minos14/janis/dae_noise_data/noise/*/*"

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/latest_checkpoint.ckpt",
                                                    save_weights_only=True, save_best_only=True,
                                                    period=1, verbose=1)
                 ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
                                         name='Adam')

    m = Model(ts_length=6001, use_bias=False, activation=None, drop_rate=0.001, channels=2, optimizer=optimizer,
              loss='mean_squared_error', callbacks=callbacks,
              dt=0.01, decimation_factor=2, cwt=True, yshape=50)
    m.build_model(filter_root=8, depth=6, fully_connected=False, max_pooling=False)
    m.summarize()
    m.train_model_generator(signal_file=signal_files, noise_file=noise_files, batch_size=40, epochs=350)
    m.save_model()
    m.plot_history()


    # XXX Fully Connected Layer has bad effect on denoising
