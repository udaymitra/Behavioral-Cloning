import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, ELU, Convolution2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
import model_reader

import image_util
from input_reader_helper import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'data/IMG/', 'The directory of the image data.')
flags.DEFINE_string('csv_path', 'data/driving_log.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')
flags.DEFINE_string('output_dir', 'training_out', 'Output directory to save model.')
flags.DEFINE_string('model', '', 'pre-trained model file path')

def main(_):
    drive_entries = read_drive_entries_from_csv(FLAGS.csv_path, FLAGS.imgs_dir)
    train_drive_entries, val_drive_entries = split_train_val(drive_entries)
    train_generator = get_keras_generator(train_drive_entries, FLAGS.batch_size, image_util.normalize_image)
    val_generator = get_keras_generator(val_drive_entries, FLAGS.batch_size, image_util.normalize_image)
    train_images_size = len(train_drive_entries) * CONFIG["num_training_entries_per_image"]
    val_images_size = len(val_drive_entries) * CONFIG["num_training_entries_per_image"]
    print("train_images_size: %d"%train_images_size)
    print("val_images_size: %d"%val_images_size)

    model = model_reader.read_model(FLAGS.model) if FLAGS.model else getNvidiaModel(FLAGS.lrate)
    model.fit_generator(train_generator,
                        samples_per_epoch = train_images_size,
                        nb_epoch=FLAGS.num_epochs,
                        validation_data=val_generator,
                        nb_val_samples=val_images_size)

    json = model.to_json()
    model.save_weights(FLAGS.output_dir + '/model.h5')
    with open(FLAGS.output_dir + '/model.json', 'w') as f:
        f.write(json)

def getNvidiaModel(learning_rate):
    ch, row, col = 3, 66, 200  # camera format
    model = Sequential([
        Conv2D(24, 5, 5, input_shape=(row, col, ch), subsample=(2, 2), border_mode='valid', activation='elu'),
        Dropout(.2),
        Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'),
        Dropout(.2),

        Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'),
        Dropout(.2),
        Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'),
        Dropout(.2),
        Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'),
        Dropout(.4),

        Flatten(),
        Dense(1164, activation='elu'),
        Dropout(.5),
        Dense(100, activation='elu'),
        Dropout(.5),

        Dense(50, activation='elu'),
        Dropout(.5),
        Dense(10, activation='elu'),
        Dropout(.5),
        Dense(1, name='output'),
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model

def getCommaAiModel(learning_rate):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Convolution2D(16, 8, 8, input_shape=(row, col, ch), subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model

def getModel(learning_rate):
    # input image is of shape 80x160x3
    model = Sequential([
        Conv2D(nb_filter=32, nb_row=6, nb_col=3, input_shape=(80, 160, 3), border_mode='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(nb_filter=64, nb_row=6, nb_col=3, border_mode='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(128, 6, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, 6, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model

if __name__ == '__main__':
    tf.app.run()