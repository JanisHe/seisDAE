import os
import glob
import pickle
import random
import numpy as np


def save_obj(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=0)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def str2bool(string):
    """
    Returns a bool from a string of "True" or "False"
    :param string: type str, {"True", "False"}
    :return: type bool
    :raise: ValueError
            If string ist not "True", "true", "False", "false"

    # >>> str2bool("True")
    # True
    # >>> str2bool("False")
    # False
    """
    if string not in ["True", "true", "False", "false"]:
        msg = "string is not True or False"
        raise ValueError(msg)
    else:
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False


def readtxt(fname):
    """
    readtxt reads a file containing parameters. "#" are comments and will be ignored. The parameter's name and its value
    are seperated by "=". First mention the name and afterwards the value. Example for a short file:

    # Settings for a test
    numbers = 12         # Gives numbers
    output  = save.pdf   # Saves the output

    The function returns a dictionary, containing parameter and its value.

    :param fname: Path and name of the file that will be read
    :type fname: str

    :return: dict
    """

    # Open file and creating empty dictionary
    fopen = open(fname, "r")
    parameters = {}
    line = fopen.readline()

    # Reading through each line in fname
    # Lines containing "#" in first values are ignored
    while line:
        if "#" not in line[:5] and "=" in line:
            param = line.split("=")          # Split between value name and rest
            name = param[0]                  # Get name of value
            value = param[1].split("#")[0]   # Split between comment and value and extract value
            name = name.strip()
            value = value.strip()            # Remove whitespace

            # Try to convert value into float
            try:
                value = float(value)
            except ValueError:
                pass

            # Otherwise value is of type string
            if isinstance(value, str) is True:
                value = value.split("\n")[0]

                # Test for None
                if value.lower() == 'none':
                    value = None

                # Try to convert value into bool
                try:
                    value = str2bool(value)
                except ValueError:
                    pass

                # Test for tuple
                try:
                    value = tuple(map(int, value.split(', ')))
                except:
                    pass

            parameters.update({name: value})
        line = fopen.readline()

    fopen.close()

    return parameters

def rms(x):
    """
    Root mean square of array x
    :param x:
    :return:
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal, noise, decibel=True):
    """
    SNR in dB
    :param signal:
    :param noise:
    :param decibel:
    :return:
    """
    if decibel is True:
        return 20 * np.log10(rms(signal) / rms(noise))
    else:
        return rms(signal) / rms(noise)


def is_nan(num):
    return num != num


def shift_array(array: np.array):
    """
    Shifts numpy array randomly to the left or right for data augmentation.
    For shift zeros are added to keep the same length.

    :param array: numpy array
    :return: shifted numpy array
    """
    shift = random.randint(-int(len(array)/2), int(len(array)/2))
    zeros = np.zeros(np.abs(shift))
    if shift > 0:
        array = np.concatenate((zeros, array))
    elif shift < 0:
        array = np.concatenate((array, zeros))

    return array


def remove_file(filename: str, verbose=False):
    """
    Function deletes filenames.
    """
    if verbose is True:
        print("File with error:", filename)
    os.remove(filename)


def check_npz(npz_filename: str, zero_check=False, verbose=False):
    """
    Trys to read npz file. If not the file is deleted
    :param npz_filename: filename to check
    :param zero_check: If True, files that contains almost zeros will be deleted.
    :param verbose: Default False. If True, files that are deleted will be printed.
    """
    try:
        dataset = np.load(npz_filename)
        data = dataset["data"]
        if zero_check is True:
            if np.max(np.abs(data)) <= 1e-15:
                remove_file(filename=npz_filename, verbose=verbose)
    except Exception:
        remove_file(filename=npz_filename, verbose=verbose)


def check_signal_zeros(signal_dir: str, extension="npz", verbose=False):
    """
    Function checks if signal files contain a file with zeros, since these files are not allowed for training.
    :param signal_dir: Full pathname for signal files
    :param extension: Extension of filenames, default is npz.
    :param verbose: Default False. If True, files that are deleted will be printed.
    """
    files = glob.glob(os.path.join(signal_dir, f"*.{extension}"))
    # Loop over all files and check whether the data can be read and the file does not contain zeros only.
    for filename in files:
        check_npz(npz_filename=filename, zero_check=True, verbose=verbose)


def check_noise_files(noise_dirname: str, extension="npz", **kwargs):
    """
    Reads all files in noise_dirname to check all .npz files.
    If a file cannot be read it is deleted.

    :param noise_dirname: Directory of noise npz files
    :param extension: Extension of filenames, default is npz.
    """
    files = glob.glob(os.path.join(noise_dirname, f"*.{extension}"))
    for filename in files:
        check_npz(npz_filename=filename, **kwargs)


def old_data_augmentation(signal_npz_file, noise_npz_file, ts_length=6001):
    signal = np.load(signal_npz_file)
    noise = np.load(noise_npz_file)
    # Read signal and noise from npz files
    try:
        p_samp = signal["itp"]  # Sample of P-arrival
        s_samp = signal["its"]  # Sample of S-arrival
    except KeyError:
        p_samp = None
        s_samp = None

    # Read data arrays from signal and noise
    signal = signal["data"]
    noise = noise["data"][:ts_length]

    # epsilon = 0  # Avoiding zeros in added arrays
    # shift1 = np.random.uniform(low=-1, high=1, size=int(self.ts_length - s_samp)) * epsilon
    # TODO: Check data augmentation if correct; Ignore itp and its
    # signal = shift_array(array=signal)
    # signal = signal[:self.ts_length]
    if p_samp and s_samp:
        if int(ts_length - s_samp) < 0:
            shift1 = np.zeros(0)
        else:
            shift1 = np.zeros(shape=int(ts_length - s_samp))
        signal = np.concatenate((shift1, signal))
        # Cut signal to length of ts_length and arrival of P-phase is included
        p_samp += len(shift1)
        s_samp += len(shift1)
        start = random.randint(0, p_samp)
        signal = signal[start:start + ts_length]
    else:  # XXX Add case just for p_samp
        if ts_length > len(signal):
            start = random.randint(0, len(signal) - ts_length - 1)
            signal = signal[start:int(start + ts_length)]
        else:
            signal = signal[:ts_length]

    return signal, noise


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import random

    # files = glob.glob("/home/janis/CODE/seismic_denoiser/example_data/signal/*")
    # d = np.load(random.choice(files))
    # data = d["data"]
    # shifted = shift_array(data)
    # plt.plot(shifted[:6001])
    # plt.show()

    files_signal = glob.glob("/home/janis/CODE/seismic_denoiser/example_data/signal/*")
    files_noise = glob.glob("/home/janis/CODE/seismic_denoiser/example_data/noise/*")
    signal = random.choice(files_signal)
    print(signal)
    s, n = old_data_augmentation(signal_npz_file=signal,
                                 noise_npz_file=random.choice(files_noise))
    d = np.load(signal)
    data = d["data"]
    shifted = shift_array(data)
    plt.plot(s, alpha=0.5)
    plt.plot(shifted[:6001], alpha=0.5)
    plt.show()
