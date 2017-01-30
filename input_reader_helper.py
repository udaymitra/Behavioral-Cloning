import csv
from os import listdir
from os.path import join
import image_util
import numpy as np

CAMERA_LEFT_RIGHTOFFSET = 0.2

class DriveLogEntry:
    def __init__(self, csv_entry, base_dir):
        assert len(csv_entry) == 7
        self.center_file_path = join(base_dir, csv_entry[0].split('/')[-1])
        self.left_file_path = join(base_dir, csv_entry[1].split('/')[-1])
        self.right_file_path = join(base_dir, csv_entry[2].split('/')[-1])

        self.center_image = image_util.read_image(self.center_file_path)
        self.left_image = image_util.read_image(self.left_file_path)
        self.right_image = image_util.read_image(self.right_file_path)

        self.steering = float(csv_entry[3])

    # return tuples of images and its steering
    # center image, left image with camera offset, right image with camera offset
    # and center image flipped right to left
    # If normalize method is passed in, then applies normalization on all the images
    def get_training_data(self, normalize_method=None):
        data = [(self.center_image, self.steering),
                (self.left_image, self.steering - CAMERA_LEFT_RIGHTOFFSET),
                (self.right_image, self.steering + CAMERA_LEFT_RIGHTOFFSET),
                (image_util.flipimage(self.center_image), -1 * self.steering)]
        if (normalize_method):
            data = [(normalize_method(entry[0]), entry[1]) for entry in data]
        return data

    def print(self):
        print(self.center_file_path)
        print(self.left_file_path)
        print(self.right_file_path)
        print(self.steering)

def read_drive_entries_from_csv(csv_path, dir_base_path):
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    drive_entries = [DriveLogEntry(row, dir_base_path) for row in reader]
    return drive_entries

def get_training_data(drive_entries, normalize_method=None):
    images = []
    labels = []
    for entry in drive_entries:
        data = entry.get_training_data(normalize_method)
        images_and_labels = [list(t) for t in zip(*data)]
        images.extend(images_and_labels[0])
        labels.extend(images_and_labels[1])
    return (images, labels)

def get_training_data_generator(images, labels, batch_size=128):
    num_examples = len(images)
    while True:
        out_images = []
        out_labels = []
        for i in range(batch_size):
            random = int(np.random.choice(num_examples, 1))
            out_images.append(images[random])
            out_labels.append(labels[random])
        yield (np.array(out_images), np.array(out_labels))