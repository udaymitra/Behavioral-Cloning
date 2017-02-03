import csv
from os import listdir
from os.path import join
from image_util import *
import numpy as np
from sklearn.utils import shuffle
import random
from config import CONFIG

class DriveLogEntry:
    def __init__(self, csv_entry, base_dir):
        assert len(csv_entry) == 7
        self.center_file_path = join(base_dir, csv_entry[0].split('/')[-1])
        self.left_file_path = join(base_dir, csv_entry[1].split('/')[-1])
        self.right_file_path = join(base_dir, csv_entry[2].split('/')[-1])

        self.center_image = read_image(self.center_file_path)
        self.left_image = read_image(self.left_file_path)
        self.right_image = read_image(self.right_file_path)

        self.steering = float(csv_entry[3])

    def get_validation_data(self, normalize_method=None):
        img = self.center_image
        steer = self.steering
        return (normalize_method(img), steer)

    def get_training_data_with_augmentation(self, normalize_method=None, augment_prob=1, bias = CONFIG["bias"]):
        data = []
        img = self.center_image
        steer = self.steering

        if np.random.rand() < augment_prob:
            # randomly choose which camera to use among (center, left, right)
            # if we picked left/right camera, adjust steer accordingly
            camera = random.choice(['center', 'left', 'right'])
            if camera == 'left':
                img = self.left_image
                steer = self.steering + CONFIG["camera_correction"]
            elif camera == 'right':
                img = self.right_image
                steer = self.steering - CONFIG["camera_correction"]

            # mirror images with probability=0.5
            if random.choice([True, False]):
                img = flipimage(img)
                steer *= -1

            # add shadow to images with probability=0.5
            if random.choice([True, False]):
                img = add_random_shadow(img)

            # slightly change steering direction
            # steer += np.random.normal(0, CONFIG['steering_augmentation_sigma'])

            # check that each element in the batch meet the condition
            # steer_magnitude_thresh = np.random.rand()
            # if (abs(steer) + bias) >= steer_magnitude_thresh:
            #     data.append((img, steer))

            # shooting for equal distribution of steering angles
            # steering_bin_size = 0.05
            # rounded_steering_val_min = int(steer / steering_bin_size) * steering_bin_size
            # rounded_steering_val_max = rounded_steering_val_min + steering_bin_size
            # random_value = np.random.rand()
            # random_value = random_value * 2 - 1
            # if (random_value >= rounded_steering_val_min and random_value < rounded_steering_val_max):

            # only valid steering angles
            if (steer <= 1.0 and steer >= -1.0):
                data.append((img, steer))
        else:
            data.append((img, steer))

        if (normalize_method):
            data = [(normalize_method(entry[0]), entry[1]) for entry in data]
        return data

    def print(self):
        print(self.center_file_path)
        print(self.left_file_path)
        print(self.right_file_path)
        print(self.steering)

    def get_steering_bin(self):
        normalized_steer = 1 + self.steering
        return int(normalized_steer / CONFIG["steering_bin_size"])

def read_drive_entries_from_csv(csv_path, dir_base_path):
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    drive_entries = [DriveLogEntry(row, dir_base_path) for row in reader]
    return drive_entries

def split_train_val(drive_entries):
    shuffle(drive_entries)
    num_train = int(CONFIG["training_data_split"] * len(drive_entries))
    return drive_entries[:num_train], drive_entries[num_train:]

def get_training_data(drive_entries, normalize_method=None):
    images = []
    labels = []
    for entry in drive_entries:
        data = entry.get_training_data(normalize_method)
        images_and_labels = [list(t) for t in zip(*data)]
        images.extend(images_and_labels[0])
        labels.extend(images_and_labels[1])
    return (images, labels)

def get_validation_generator(drive_entries, batch_size, normalize_method=None):
    num_examples = len(drive_entries)
    while True:
        entry_idx = 0
        while entry_idx < num_examples:
            out_images = []
            out_labels = []
            num_batch_examples = 0
            while num_batch_examples < batch_size and entry_idx < num_examples:
                drive_entry = drive_entries[entry_idx]
                data = drive_entry.get_validation_data(normalize_method)
                num_batch_examples += 1
                out_images.append(data[0])
                out_labels.append(data[1])
            yield (np.array(out_images), np.array(out_labels))

def get_training_data_generator_equal_steering_distribution(drive_entries, batch_size, augment_prob=1,
                                                            bias=CONFIG["bias"], normalize_method=None):
    # split entries into bins
    steering_bin_entry_dict = {}
    for entry in drive_entries:
        bin = entry.get_steering_bin()
        if bin in steering_bin_entry_dict:
            steering_bin_entry_dict[bin].append(entry)
        else:
            steering_bin_entry_dict[bin] = [entry]

    num_batch_examples = 0
    out_images = []
    out_labels = []
    while True:
        for bin_number, bin_entries in steering_bin_entry_dict.items():
            entry_index = random.randint(0, len(bin_entries) - 1)
            entry = bin_entries[entry_index]
            data = entry.get_training_data_with_augmentation(normalize_method, augment_prob=augment_prob, bias=bias)
            num_batch_examples += len(data)
            if data:
                images_and_labels = [list(t) for t in zip(*data)]
                out_images.extend(images_and_labels[0])
                out_labels.extend(images_and_labels[1])
            if (num_batch_examples >= batch_size):
                yield (np.array(out_images), np.array(out_labels))
                # reset batch vars
                num_batch_examples = 0
                out_images = []
                out_labels = []

def get_training_data_generator(drive_entries, batch_size, augment_prob=1, bias = CONFIG["bias"], normalize_method=None):
    num_examples = len(drive_entries)
    while True:
        entry_idx = 0
        while entry_idx < num_examples:
            out_images = []
            out_labels = []
            num_batch_examples = 0
            while num_batch_examples < batch_size and entry_idx < num_examples:
                drive_entry = drive_entries[entry_idx]
                data = drive_entry.get_training_data_with_augmentation(normalize_method,
                                                                       augment_prob=augment_prob, bias=bias)
                num_batch_examples += len(data)
                if data:
                    images_and_labels = [list(t) for t in zip(*data)]
                    out_images.extend(images_and_labels[0])
                    out_labels.extend(images_and_labels[1])
                entry_idx += 1
            yield (np.array(out_images), np.array(out_labels))

# get rid of this generator below. only used for visualization purposes
def get_generator(drive_entries, batch_size, normalize_method=None):
    num_examples = len(drive_entries)
    entry_idx = 0
    while entry_idx < num_examples:
        out_images = []
        out_labels = []
        num_batch_examples = 0
        while num_batch_examples < batch_size and entry_idx < num_examples:
            drive_entry = drive_entries[entry_idx]
            data = drive_entry.get_training_data_with_augmentation(normalize_method)
            num_batch_examples += len(data)
            images_and_labels = [list(t) for t in zip(*data)]
            out_images.extend(images_and_labels[0])
            out_labels.extend(images_and_labels[1])
            entry_idx += 1
        yield (np.array(out_images), np.array(out_labels))
