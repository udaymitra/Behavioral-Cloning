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
    model = model_reader.read_model(FLAGS.model) if FLAGS.model else getNvidiaModel(FLAGS.lrate)
    model.compile(optimizer=Adam(), loss='mse')

    # for (lrate, aug_prob) in [(FLAGS.lrate, 0),
    #                           (0.8 * FLAGS.lrate, 0.5),
    #                           (0.5 * FLAGS.lrate, 0.8),
    #                           (0.3 * FLAGS.lrate, 1)]:
    #     train_drive_entries, val_drive_entries = split_train_val(drive_entries)
    #     train_model(model, train_drive_entries, val_drive_entries, augment_prob=aug_prob,
    #                 num_training_entries_per_image = CONFIG["num_training_entries_per_image"])

    num_training_entries_per_image = CONFIG["num_training_entries_per_image"]
    # train_model(model, drive_entries, drive_entries, augment_prob_threshold=0,
    #                             num_training_entries_per_image=num_training_entries_per_image, keep_pr_threshold=0.5)
    # train_model(model, drive_entries, drive_entries, augment_prob_threshold=0,
    #                             num_training_entries_per_image=num_training_entries_per_image, keep_pr_threshold=0.9)
    # train_model(model, drive_entries, drive_entries, augment_prob_threshold=0,
    #                             num_training_entries_per_image=num_training_entries_per_image, keep_pr_threshold=0)

    train_model(model, drive_entries, drive_entries, augment_prob_threshold=0,
                                num_training_entries_per_image=num_training_entries_per_image, keep_pr_threshold=0)

    json = model.to_json()
    print("saving model to file")
    model.save_weights(FLAGS.output_dir + '/model.h5')
    with open(FLAGS.output_dir + '/model.json', 'w') as f:
        f.write(json)

def train_model(model, train_drive_entries, val_drive_entries, augment_prob_threshold=0,
                num_training_entries_per_image = CONFIG["num_training_entries_per_image"], keep_pr_threshold = 0.5):
    # train_generator = get_training_data_generator(train_drive_entries,
    #                                             FLAGS.batch_size,
    #                                             augment_prob_threshold=augment_prob_threshold,
    #                                             normalize_method=image_util.normalize_image,
    #                                             keep_pr_threshold = keep_pr_threshold)

    train_generator = get_training_data_generator_equal_steering_distribution(train_drive_entries,
                                                  FLAGS.batch_size,
                                                  augment_prob_threshold=augment_prob_threshold,
                                                  normalize_method=image_util.normalize_image,
                                                  keep_pr_threshold=keep_pr_threshold)

    val_generator = get_validation_generator(val_drive_entries, FLAGS.batch_size, image_util.normalize_image)
    train_images_size = len(train_drive_entries) * num_training_entries_per_image
    train_images_size = (int(train_images_size / FLAGS.batch_size) + 1) * FLAGS.batch_size
    val_images_size = len(val_drive_entries)
    val_images_size = ((int(val_images_size) / FLAGS.batch_size) +1) * FLAGS.batch_size
    print("train_images_size: %d" % train_images_size)
    print("val_images_size: %d" % val_images_size)
    model.fit_generator(train_generator,
                        samples_per_epoch=train_images_size,
                        nb_epoch=FLAGS.num_epochs,
                        validation_data=val_generator,
                        nb_val_samples=val_images_size)

def getNvidiaModel(learning_rate):
    ch, row, col = 3, 66, 200  # camera format
    model = Sequential([
        Conv2D(24, 5, 5, input_shape=(row, col, ch), subsample=(2, 2), border_mode='valid', activation='elu'),
        # Dropout(.2),
        Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'),
        # Dropout(.2),

        Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'),
        # Dropout(.2),
        Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'),
        # Dropout(.2),
        Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'),
        # Dropout(.4),

        Flatten(),
        Dense(1164, activation='elu'),
        # Dropout(.2),
        Dense(100, activation='elu'),
        # Dropout(.2),

        Dense(50, activation='elu'),
        # Dropout(.2),
        Dense(10, activation='elu'),
        Dense(1, name='output'),
    ])
    return model

if __name__ == '__main__':
    tf.app.run()
    import gc; gc.collect()