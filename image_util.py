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
    resize = resize_image(image, resize_shape=(200, 66))
    return make_zero_mean(resize)

# This function is to generate flipped images to simulate opposite side driving
def flipimage(image):
    return cv2.flip(image, 1)

# sample_image_path = '/code/carnd/behavior cloning sims/data2/IMG/center_2017_01_27_15_26_38_557.jpg'
# img = read_image(sample_image_path)
#
# f1 = plt.figure()
# f2 = plt.figure()
# ax1 = f1.add_subplot(111)
# ax1.plot(range(0,10))
# ax2 = f2.add_subplot(111)
# ax2.plot(range(10,20))
# plt.show()

# plt.subplot(2,1,1)
# plt.imshow(img)
# plt.subplot(2,1,2)
# plt.imshow(resize_image(img, (200, 66)))
# plt.show()