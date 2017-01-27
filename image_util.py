import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

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

def normalize_image(image):
    # gray = convert_to_gray(image)
    resize = resize_image(image)
    return make_zero_mean(resize)

# sample_image_path = '/code/carnd/behavior cloning sims/IMG/center_2017_01_26_20_22_58_395.jpg'
# img = read_image(sample_image_path)
# plt.imshow(img)
# plt.show()
# plt.imshow(resize_image(img))
# plt.show()