import glob
import math
import os

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model, Model
import training.train_functions.metrics
from training.train_functions.metrics import *

#K.set_image_dim_ordering('th')


def get_last_model_path(model_file_path):
    model_names = glob.glob(model_file_path + '*.h5')
    if not model_names:
      model_names = glob.glob(model_file_path + '*.hdf5')
    return sorted(model_names, key=os.path.getmtime)[-1]

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


def get_callbacks(ex, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()

    callbacks.append(ModelCheckpoint(filepath=os.path.join(ex.observers[0].dir, 'epoch_{epoch:03d}-loss{val_loss:.3f}_model.hdf5'),
                                     save_best_only=True, verbose=verbosity, monitor='val_loss'))
    callbacks.append(CSVLogger(os.path.join(ex.observers[0].dir, 'metrics.csv')))
   # callbacks.append(CSVLogger(os.path.join(ex.observers[0].dir, 'metrics.json')))

 #   callbacks.append(TensorBoard(log_dir=ex.observers[0].dir))

    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def load_old_model(model_file, verbose=True):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'vod_coefficient': vod_coefficient,
                      'vod_coefficient_loss': vod_coefficient_loss,
                      'focal_loss': focal_loss,
                      'focal_loss_fixed': focal_loss,
                      'dice_and_xent': dice_and_xent}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        if verbose:
            print('Loading model from {}...'.format(model_file))
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

def load_old_model(model_file, verbose=True, config=None) -> Model:
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'vod_coefficient': vod_coefficient,
                      'vod_coefficient_loss': vod_coefficient_loss,
                      'focal_loss': focal_loss,
                      'focal_loss_fixed': focal_loss,
                      'dice_and_xent': dice_and_xent,
                      'double_dice_loss': double_dice_loss,
                      'dice_distance_weighted_loss': dice_distance_weighted_loss,
                      'loss': dice_distance_weighted_loss(tf.keras.backend.zeros(1))
                      }
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        if verbose:
            print('Loading model from {}...'.format(model_file))
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        print(error)
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            if config is not None:
                print('Trying to build model manually...')
                loss_func = getattr(training.train_functions.metrics, config['loss'])
                model_func = getattr(training.model, config['model_name'])
                model = model_func(input_shape=config["input_shape"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   **{'dropout_rate': config['dropout_rate'],
                                      'loss_function': loss_func,
                                      'mask_shape': None if config["weight_mask"] is None else config["input_shape"],
                                      # TODO: change to output shape
                                      'old_model_path': config['old_model']})
                model.load_weights(model_file)
                return model
            else:
                raise


def train_model(model, ex, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None, output_folder='.'):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return:
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        max_queue_size=20,
                        workers=1,
                        use_multiprocessing=False,
                        callbacks=get_callbacks(ex=ex,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))
