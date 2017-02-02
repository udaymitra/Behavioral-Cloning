import matplotlib.pyplot as plt
import input_reader_helper
import math
import numpy as np
from sklearn.utils import shuffle

def visualize_steering_distribution():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./data/driving_log.csv", "data/IMG")
    steering_list = [entry.steering for entry in drive_entries]
    visualize_steering(steering_list, 'Steering angle distribution in data')

def visualize_steering(steering_list, title):
    plt.title(title)
    plt.hist(steering_list, 100, normed=0, facecolor='green', alpha=0.75)
    plt.ylabel('# images'), plt.xlabel('steering angle')
    plt.show()

def show_images(images, titles, cols):
    assert len(images) == len(titles)
    num_images = len(images)
    rows = math.ceil(num_images/cols)
    fig, axes = plt.subplots(rows, cols)
    # fig.set_figheight(rows * 5)
    # fig.set_figwidth(cols * 5)
    idx = 0
    for i, ax in enumerate(axes.flat):
        if idx < num_images:
            ax.imshow(images[idx])
            ax.set_xlabel(titles[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            idx = idx + 1
    plt.show()

def show_sample_images():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./sample_driving_log.csv", "data/IMG")
    assert (len(drive_entries) == 3)
    sample_images = []
    titles = []
    for entry in drive_entries:
        sample_images.append(entry.center_image)
        titles.append("center image. steering: %f"%entry.steering)
        sample_images.append(entry.left_image)
        titles.append("left image. corrected steering: %f" % (entry.steering + 0.2))
        sample_images.append(entry.right_image)
        titles.append("right image. corrected steering: %f" % (entry.steering - 0.2))
    show_images(sample_images, titles, 3)

def show_sample_images_after_augmentation():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./sample_driving_log.csv", "data/IMG")
    assert (len(drive_entries) == 3)
    sample_images = []
    titles = []
    data_generator = input_reader_helper.get_generator(drive_entries, 10)
    for d in data_generator:
        sample_images = sample_images + list(d[0])
        titles += [("steering: %f" % steer) for steer in list(d[1])]

    show_images(sample_images, titles, 3)

def visualize_steering_distribution(title, augmentation_prob=1):
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./data/driving_log.csv", "data/IMG")
    data_generator = input_reader_helper.get_keras_generator(drive_entries, 512, augment_prob=augmentation_prob)
    steering_list = []
    num_entries = 0
    while num_entries < len(drive_entries):
        imgs, steerings =  next(data_generator)
        steering_list += list(steerings)
        num_entries += steerings.shape[0]

    print("num training data: %d" % len(steering_list) )
    visualize_steering(steering_list, title)

def visualize_bias_parameter_effect():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./data/driving_log.csv", "data/IMG")
    biases = [0, 0.2, 0.4, 0.6, 0.8, 1]
    fig, axarray = plt.subplots(len(biases))
    plt.suptitle('Effect of bias parameter on steering angle distribution', fontsize=14, fontweight='bold')
    for i, ax in enumerate(axarray.ravel()):
        b = biases[i]
        data_generator = input_reader_helper.get_keras_generator(drive_entries, 512, augment_prob=1, bias=b)
        steering_list = []
        num_entries = 0
        while num_entries < 2 * len(drive_entries):
            imgs, steerings = next(data_generator)
            steering_list += list(steerings)
            num_entries += steerings.shape[0]

        ax.hist(steering_list, 50, normed=1, facecolor='green', alpha=0.75)
        ax.set_title('Bias: {:02f}'.format(b))
        ax.axis([-1., 1., 0., 2.])
    plt.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)
    plt.show()

plt.close("all")
# visualize_steering_distribution('Steering angle distribution after augmentation', augmentation_prob=1)
visualize_bias_parameter_effect()