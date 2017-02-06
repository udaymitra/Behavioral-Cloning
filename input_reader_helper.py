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

        self.steering = float(csv_entry[3])

    def get_center_image(self):
        return read_image(self.center_file_path)

    def get_left_image(self):
        return read_image(self.left_file_path)

    def get_right_image(self):
        return read_image(self.right_file_path)

    def get_validation_data(self, normalize_method=None):
        img = self.get_center_image()
        steer = self.steering
        return (normalize_method(img), steer)

    def get_training_data_with_augmentation_no_normalization(self, augment_prob_threshold, keep_pr_threshold):
        img = self.get_center_image()
        steer = self.steering

        if np.random.rand() < augment_prob_threshold:
            return (img, steer)

        # Augment image logic below

        # randomly choose which camera to use among (center, left, right)
        # if we picked left/right camera, adjust steer accordingly
        camera = random.choice(['center', 'left', 'right'])
        if camera == 'left':
            img = self.get_left_image()
            steer = self.steering + CONFIG["camera_correction"]
        elif camera == 'right':
            img = self.get_right_image()
            steer = self.steering - CONFIG["camera_correction"]

        # mirror images with probability=0.5
        if random.choice([True, False]):
            img = flipimage(img)
            steer *= -1

        # add shadow to images with probability=0.5
        if random.choice([True, False]):
            img = add_random_shadow(img)

        # adjust brightness with probability=0.5
        if random.choice([True, False]):
            img = set_random_brightness(img)

        # If steer angle is straight (close to 0), keep only small portion of that data
        # This is limited by the value set for keep_pr_threshold
        if abs(steer) > 0.1 or (abs(steer) <= 0.1 and np.random.rand() > keep_pr_threshold):
            return (img, steer)

    def get_training_data_with_augmentation(self, normalize_method=None, augment_prob_threshold=0, keep_pr_threshold=0.5):
        data = self.get_training_data_with_augmentation_no_normalization(augment_prob_threshold, keep_pr_threshold)
        return (normalize_method(data[0]), data[1]) if normalize_method else data

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

def get_training_data_generator_equal_steering_distribution(drive_entries, batch_size,
                                                            augment_prob_threshold=0,
                                                            keep_pr_threshold=0.5,
                                                            normalize_method=None):
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
            data = entry.get_training_data_with_augmentation(normalize_method, augment_prob_threshold=augment_prob_threshold,
                                                             keep_pr_threshold=keep_pr_threshold)
            if data:
                num_batch_examples += 1
                out_images.append(data[0])
                out_labels.append(data[1])
            if (num_batch_examples >= batch_size):
                yield (np.array(out_images), np.array(out_labels))
                # reset batch vars
                num_batch_examples = 0
                out_images = []
                out_labels = []

def get_training_data_generator(drive_entries, batch_size, augment_prob_threshold=0,
                                keep_pr_threshold=0.5,
                                normalize_method=None):
    num_examples = len(drive_entries)
    out_images = []
    out_labels = []
    num_batch_examples = 0
    while True:
        entry_idx = np.random.randint(0, num_examples - 1)
        drive_entry = drive_entries[entry_idx]
        data = drive_entry.get_training_data_with_augmentation(normalize_method, augment_prob_threshold=augment_prob_threshold,
                                                               keep_pr_threshold=keep_pr_threshold)
        if data:
            num_batch_examples += 1
            out_images.append(data[0])
            out_labels.append(data[1])

        if num_batch_examples >= batch_size:
            yield (np.array(out_images), np.array(out_labels))
            out_images = []
            out_labels = []
            num_batch_examples = 0