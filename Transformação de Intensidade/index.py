import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

enhance_me = Image.open('enhance-me.gif')
enhance_me = enhance_me.convert('L')

# apply median filter in image
enhance_me = enhance_me.filter(ImageFilter.MedianFilter(5))

fig_3_8 = Image.open('Fig0308.tif')
fig_3_8 = fig_3_8.convert('L')


def plot_result(original, transformed, *args):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(transformed, cmap='gray')
    axs[1].set_title('Transformed with {}'.format(args))


def process(img, transformation, *args):
    transformed = transformation(img, *args)
    plot_result(img, transformed, *args)

#Exercicio 1

def apply_log_transformation(img, c):
    img = np.array(img)
    img = c * np.log2(1 + img)
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img


for value in [1, 10, 100, 1000]:
    process(enhance_me, apply_log_transformation, value)

#Exercicio 2

def apply_gamma_transformation(img, c, gamma):
    img = np.array(img)
    img = c * np.power(img, gamma)
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img


for value in [0.1, 0.5, 1, 2, 5, 10]:
    process(enhance_me, apply_gamma_transformation, 1, value)

for value in [0.1, 0.5, 1, 2, 5, 10]:
    process(fig_3_8, apply_gamma_transformation, 1, value)

#Exercicio 3

def get_bit_plane(img, bit):
    img = np.array(img)

    # extract n bit layer
    img = np.bitwise_and(img, 2 ** bit)

    # normalize
    img = img / 2 ** bit

    return img


plt.figure(figsize=(15, 15))

enhance_me = np.array(enhance_me)
enhance_me = (enhance_me - enhance_me.min()) / (enhance_me.max() - enhance_me.min()) * 255
enhance_me = Image.fromarray(enhance_me.astype(np.uint8))

plt.subplot(3, 3, 1)
plt.imshow(enhance_me, cmap='gray')
plt.title('Original')

for bit in range(8):
    plt.subplot(3, 3, bit + 2)
    plt.imshow(get_bit_plane(enhance_me, bit), cmap='gray')
    plt.title('Bit plane {}'.format(bit))

plt.show()

plt.figure(figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(fig_3_8, cmap='gray')
plt.title('Original')

for bit in range(8):
    plt.subplot(3, 3, bit + 2)
    plt.imshow(get_bit_plane(fig_3_8, bit), cmap='gray')
    plt.title('Bit plane {}'.format(bit))

#Exercicio 4

from PIL import ImageOps


def equalize_histogram(img):
    img1 = np.array(img)

    plt.figure(figsize=(50, 50))
    plt.rcParams.update({'font.size': 32})

    plt.subplot(2, 2, 1)
    plt.hist(img1.ravel(), bins=256, range=(0, 255))
    plt.title('Original' )

    img = ImageOps.equalize(img)
    img = np.array(img)

    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), bins=256,  range=(0, 255))
    plt.title('Equalized')

    plt.show()


equalize_histogram(enhance_me)
equalize_histogram(fig_3_8)