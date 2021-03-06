import matplotlib.pyplot as plt
import input_reader_helper
import math
import image_util

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
        sample_images.append(entry.get_center_image())
        titles.append("center image. steering: %.3f"%entry.steering)

        sample_images.append(entry.get_left_image())
        titles.append("left image. corrected steering: %.3f" % (entry.steering + 0.2))

        sample_images.append(entry.get_right_image())
        titles.append("right image. corrected steering: %.3f" % (entry.steering - 0.2))

    sample_images = [image_util.crop_image(im) for im in sample_images]
    show_images(sample_images, titles, 3)

def show_sample_images_after_augmentation():
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./sample_driving_log.csv", "data/IMG")
    assert (len(drive_entries) == 3)
    titles = []
    data_generator = input_reader_helper.get_training_data_generator(drive_entries, 12)
    sample_images, steer = next(data_generator)

    sample_images = list(sample_images)
    sample_images = [image_util.crop_image(im) for im in sample_images]
    steer = list(steer)

    titles += [("steering: %f" % st) for st in steer]
    show_images(sample_images, titles, 3)

def visualize_steering_distribution(title, augment_prob_threshold=0, keep_pr_threshold=0.8):
    drive_entries = input_reader_helper.read_drive_entries_from_csv("./data/driving_log.csv", "data/IMG")
    data_generator = input_reader_helper.get_training_data_generator(drive_entries, 512, augment_prob_threshold=augment_prob_threshold, keep_pr_threshold=keep_pr_threshold)
    steering_list = []
    num_entries = 0
    while num_entries < len(drive_entries):
        imgs, steerings = next(data_generator)
        steering_list += list(steerings)
        num_entries += steerings.shape[0]

    print("num training data: %d" % len(steering_list) )
    visualize_steering(steering_list, title)

visualize_steering_distribution('Steering angle distribution (keep_pr_threshold=0)', augment_prob_threshold=0, keep_pr_threshold=0)
# show_sample_images()
# show_sample_images_after_augmentation()

