import os
import sys

from tensorflow import keras

from model import retrain
from utils import readtxt


def main_retraining(parfile):
    """
    Main function to read parameters from file to start retraining
    If script is run as main, the default retrain_parfile is read to retrain the gr_mixed_stft model.
    :param parfile: full pathname of parameter file
    """
    print(f"Reading parameters from {parfile}", flush=True)
    parameters = readtxt(parfile)

    # Create callbacks
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=parameters['patience'], verbose=1),
                 keras.callbacks.ModelCheckpoint(filepath="./checkpoints/{}/latest_checkpoint.ckpt".
                                                 format(parameters['filename']),
                                                 save_weights_only=True, save_best_only=True,
                                                 period=1, verbose=1)
                 ]

    # Start retraining
    retrain(model_filename=parameters['model'], config_filename=parameters['config'],
            signal_pathname=parameters["signal_pathname"], noise_pathname=parameters["noise_pathname"],
            num_data=int(parameters["num_signals"]), batch_size=int(parameters['batch_size']),
            epochs=int(parameters['epochs']), validation_split=parameters['validation_split'],
            workers=int(parameters['workers']), callbacks=callbacks, filename=parameters['filename'],
            verbose=int(parameters['verbose']))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        msg = "Argument for parameters is missing. Run example."
        parfile = "./retrain_parfile"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Start to run training with parameters from parfile
    main_retraining(parfile=parfile)
