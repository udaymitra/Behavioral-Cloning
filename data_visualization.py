import matplotlib.pyplot as plt
import input_reader_helper
import math
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
        titles.append("left image. corrected steering: %f" % (entry.steering - 0.2))
        sample_images.append(entry.right_image)
        titles.append("right image. corrected steering: %f" % (entry.steering + 0.2))
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

def visualize_steering_distribution_after_augmentation():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./data/driving_log.csv", "data/IMG")
    data_generator = input_reader_helper.get_generator(drive_entries, 512)
    steering_list = []
    for d in data_generator:
        steering_list += list(d[1])
    print("num training data: %d" % len(steering_list) )
    visualize_steering(steering_list, 'Steering angle distribution after augmentation')

show_sample_images_after_augmentation()