import matplotlib.image as mpimg
import cv2
import numpy as np

def read_image(path):
    image = mpimg.imread(path)
    return image

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# original image is 160H x 320W. Reduce size
def resize_image(image, resize_shape=(160, 80)):
    return cv2.resize(image, resize_shape, interpolation=cv2.INTER_CUBIC)

# normalize values from [-0.5, 0.5]
def make_zero_mean(image):
    return (image - 128.5) / 255.

# crop out top (sky) and bottom part of the image
def crop_image(image):
    return image[60:140,:,:]

def normalize_image(image):
    cropped_image = crop_image(image)
    resize = resize_image(cropped_image, resize_shape=(200, 66))
    return make_zero_mean(resize)

# This function is to generate flipped images to simulate opposite side driving
def flipimage(image):
    return cv2.flip(image, 1)

# This function is used to augment the data
# and add random shadows to the image
def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image