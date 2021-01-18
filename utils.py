import pickle

def save_obj(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=0)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_dict_to_txt(dictionary, filename):
    f = open(filename, "w")
    for key in dictionary:
        f.write("{} = {}\n".format(key, dictionary[key]))
    f.close()


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
            param = line.split()
            parameters.update({param[0]: param[2]})
        line = fopen.readline()

    fopen.close()

    return parameters



if __name__ == "__main__":
    d = dict(loss="mse", cwt=True, ts_length=6001, kwargs=dict(yshape=80, wavelet="morlet"))
    #write_dict_to_txt(d, "./Models/settings.txt")
    save_obj(d, "./Models/settings.txt")

    #c = readtxt("./Models/settings.txt")
    c = load_obj("./Models/settings.txt")
    cwt = c['cwt']
    kwargs = c['kwargs']
    print(c)
