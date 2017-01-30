import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, ELU, Convolution2D
from keras.optimizers import Adam
from sklearn.utils import shuffle

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

def main(_):
    drive_entries = read_drive_entries_from_csv(FLAGS.csv_path, FLAGS.imgs_dir)
    (images, labels) = get_training_data(drive_entries, image_util.normalize_image)

    # split to train(90%) and validation(10%)
    images, labels = shuffle(images, labels)
    num_train_examples = int(len(images) * 0.9)
    train_images = images[:num_train_examples]
    train_labels = labels[:num_train_examples]
    val_images = images[num_train_examples:]
    val_labels = images[num_train_examples:]

    print("images: %d"%len(images))
    print("train_images: %d"%len(train_images))
    print("val_images: %d" % len(val_images))

    train_generator = get_training_data_generator(train_images, train_labels, FLAGS.batch_size)
    val_generator = get_training_data_generator(val_images, val_labels, FLAGS.batch_size)

    model = getCommaAiModel()
    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')
    model.fit_generator(train_generator, samples_per_epoch = len(train_images), nb_epoch=10,
                        validation_data=val_generator, nb_val_samples=len(val_images))

    json = model.to_json()
    model.save_weights(FLAGS.output_dir + '/model.h5')
    with open(FLAGS.output_dir + '/model.json', 'w') as f:
        f.write(json)

def getCommaAiModel():
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

    return model

def getModel():
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
    return model

if __name__ == '__main__':
    tf.app.run()