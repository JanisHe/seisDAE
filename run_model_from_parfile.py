import sys
import os
import shutil
import glob
import warnings
import tensorflow as tf

from model import Model
from utils import readtxt

if len(sys.argv) <= 1:
    msg = "Argument for parameters file is not there. Run e.g. by 'bash train.sh model_parfile'"
    raise RuntimeError(msg)
elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
    msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
    raise FileNotFoundError(msg)
else:
    # Read parfile
    parfile = sys.argv[1]
    parameters = readtxt(parfile)

    # Make copy of parfile and rename it by filename given in parameters
    shutil.copyfile(src=parfile, dst="./model_parfiles/{}.parfile".format(parameters['filename']))

    # Set up everthing to start training of model
    # Setup for GPU
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters['num_gpu'])
    except KeyError:
        warnings.warn("Run on GPU 0 because 'num_gpu' is not defined!")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Setup for signal and noise files
    signal_files = glob.glob(parameters['signal_pathname'])
    noise_files = parameters['noise_pathname']
    try:
        signal_files = signal_files[:int(parameters['num_signals'])]
    except KeyError:
        pass

    # Create callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=parameters['patience'], verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/{}/latest_checkpoint.ckpt".
                                                    format(parameters['filename']),
                                                    save_weights_only=True, save_best_only=True,
                                                    period=1, verbose=1)
                 ]

    # Create optimizer  XXX Add more optimizers and set options
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
                                         name='Adam')

    # Set kernel regularizer
    kernel_regularizer = tf.keras.regularizers.L2(0)

    # Create Model
    # 1. Read all needed parameters or generate them if not available
    try:
        drop_rate = parameters['drop_rate']
    except KeyError:
        warnings.warn("Set drop_rate = 0.1, otherwise add to parameter file")
        drop_rate = 0.1

    try:
        loss = parameters['loss_function']
    except KeyError:
        warnings.warn("Set loss function to 'binary_crossentropy', otherwise specify in parameter file!")
        loss = "binary_crossentropy"

    try:
        decimation_factor = parameters['decimation_factor']
    except KeyError:
        decimation_factor = None

    if parameters['cwt'] is True:
        dimension_kwargs = dict(cwt=True, yshape=parameters['yshape'])
    elif parameters['cwt'] is False:
        dimension_kwargs = dict(cwt=False, nfft=parameters['nfft'], nperseg=parameters['nperseg'])
    else:
        raise ValueError("cwt must be True or False!")

    try:
        filter_root = int(parameters['filter_root'])
    except KeyError:
        warnings.warn("Set filter_root = 8, otherwise specify in parameter file!")
        filter_root = 8

    try:
        depth = int(parameters['depth'])
    except KeyError:
        warnings.warn("Set depth of model to 6, otherwise specify in parameter file!")
        depth = 6

    try:
        strides = parameters['strides']
    except KeyError:
        warnings.warn('Set strides to (2, 2), otherwise specify in parameter file!')
        strides = (2, 2)

    try:
        verbose = parameters['verbose']
    except KeyError:
        verbose = 1

    try:
        workers = int(parameters['workers'])
    except KeyError:
        warnings.warn("Set number of workers to 1. Otherwise specify in parameter file!")
        workers = 1

    if workers > 1:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    try:
        max_queue_size = int(parameters['max_queue_size'])
    except KeyError:
        max_queue_size = 10

    # 2. Create Model
    m = Model(ts_length=int(parameters['ts_length']), use_bias=False, activation=None, drop_rate=drop_rate,
              channels=2, optimizer=optimizer, loss=loss, callbacks=callbacks, shuffle=True,
              validation_split=parameters['validation_split'], dt=parameters['dt'],
              decimation_factor=decimation_factor, **dimension_kwargs)

    # 3. Build model
    m.build_model(filter_root=filter_root, depth=depth, fully_connected=False, max_pooling=False,
                  strides=strides, kernel_regularizer=kernel_regularizer)

    # Print summarized model
    m.summarize()

    # 4. Start training
    m.train_model_generator(signal_file=signal_files, noise_file=noise_files,
                            batch_size=int(parameters['batch_size']), epochs=int(parameters["epochs"]),
                            workers=workers, max_queue_size=max_queue_size,
                            use_multiprocessing=use_multiprocessing, verbose=int(parameters['verbose']))

    # Save model and plot ist history
    m.save_model(filename=parameters["filename"])
    m.plot_history(filename=parameters["filename"])