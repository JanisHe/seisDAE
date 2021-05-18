import pickle
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


def signal_to_noise_ratio(signal, noise):
    """
    SNR in dB
    :param signal:
    :param noise:
    :return:
    """
    return 10 * np.log10(rms(signal) / rms(noise))


if __name__ == "__main__":
    para = readtxt("model_parfile")
    print()

