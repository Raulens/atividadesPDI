import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#Exercicio 1 - a)
def make_square(shape: (int, int), inner_shape: (int, int, int), translate: (int, int) = (0, 0)) -> np.array:
    if inner_shape[0] > shape[0] or inner_shape[1] > shape[1]:
        raise ValueError('shape_center must be inside shape')

    square = np.zeros(shape)
    x_center = shape[0] / 2 + translate[0]
    y_center = shape[1] / 2 + translate[1]

    square[int(x_center - inner_shape[0] / 2):int(x_center + inner_shape[0] / 2),
    int(y_center - inner_shape[1] / 2):int(y_center + inner_shape[1] / 2)] = 255

    pillow = Image.fromarray(square)
    rotated = pillow.rotate(inner_shape[2])
    square = np.array(rotated)

    return square


square_shape = (512, 512)
square_center_shape = (64, 64, 180)


def get_default_square():
    return make_square(square_shape, square_center_shape)


plt.imshow(make_square(square_shape, square_center_shape), cmap='gray')
plt.show()


# Exercicio 1 - b)

def fourier_transform_unshift(img):
    dft = np.fft.fft2(img)

    fourier = 20 * np.log(np.abs(dft) + 1e-8)
    phase = np.angle(dft)

    return fourier, phase


square_fourier_unshift, square_phase_unshift = fourier_transform_unshift(get_default_square())

plt.imshow(square_fourier_unshift, cmap='gray')
plt.show()

# Exercicio 1 - c)
plt.imshow(square_phase_unshift, cmap='gray')
plt.show()

#Exercicio 1 - d)

def fourier_transform(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    fourier = 20 * np.log(np.abs(dft_shift) + 1e-8)
    phase = np.angle(dft_shift)

    return fourier, phase


square_fourier, _ = fourier_transform(get_default_square())
plt.imshow(square_fourier, cmap='gray')
plt.show()

#Exercicio 1 - e)
square_center_shape = (64, 64, 40)

_, axs = plt.subplots(1, 3, figsize=(15, 15))

axs[0].imshow(make_square(square_shape, square_center_shape), cmap='gray')
axs[0].set_title('Imagem original')

square_fourier_rotated, square_phase_rotated = fourier_transform(make_square(square_shape, square_center_shape))

axs[1].imshow(square_fourier_rotated, cmap='gray')
axs[1].set_title('Espectro de Fourier')

axs[2].imshow(square_phase_rotated, cmap='gray')
axs[2].set_title('Fase')

plt.show()

#Exercicio 1 - f)

translate = (100, 100)
inner_shape = (64, 64, 0)

square_translated = make_square(square_shape, inner_shape, translate)

_, axs = plt.subplots(1, 3, figsize=(15, 15))

axs[0].imshow(square_translated, cmap='gray')
axs[0].set_title('Imagem original')

square_fourier_translated, square_phase_translated = fourier_transform(square_translated)

axs[1].imshow(square_fourier_translated, cmap='gray')
axs[1].set_title('Espectro de Fourier')

axs[2].imshow(square_phase_translated, cmap='gray')
axs[2].set_title('Fase')

plt.show()

#Exercicio 1 - g)

zoom = 3
square_center_shape = (64 * zoom, 64 * zoom, 0)

_, axs = plt.subplots(1, 3, figsize=(15, 15))

axs[0].imshow(make_square(square_shape, square_center_shape), cmap='gray')
axs[0].set_title('Imagem original')

square_fourier_zoomed, square_phase_zoomed = fourier_transform(make_square(square_shape, square_center_shape))

axs[1].imshow(square_fourier_zoomed, cmap='gray')
axs[1].set_title('Espectro de Fourier')

axs[2].imshow(square_phase_zoomed, cmap='gray')
axs[2].set_title('Fase')

plt.show()

#Exercicio 1 - h)
#Rotação: Quando uma imagem é rotacionada, sua Transformada de Fourier também é rotacionada.
#Translação: Quando uma imagem é transladada, isso resulta em uma multiplicação por uma exponencial na Transformada de Fourier.
#Zoom: Quando um zoom é aplicado a uma imagem, isso causa uma convolução na Transformada de Fourier.

#Exercicio 2 - a)

def inverse_fourier_transform(img, phase):
    # Compute inverse Fourier Transform
    magnitude_spectrum = np.exp((img - 1e-8) / 20)
    real = magnitude_spectrum * np.cos(phase)
    imag = magnitude_spectrum * np.sin(phase)
    dft_shift = real + imag * 1j
    dft = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(dft)

    return img_back

def dij(shape, i, j):
    return np.sqrt((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)

def low_pass_ideal_filter(img, cutoff):
    H = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Dij = dij(img.shape, i, j)
            if Dij <= cutoff:
                H[i, j] = 1

    return H

def low_pass_butterworth_filter(img, cutoff, order):
    H = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Dij = dij(img.shape, i, j)
            H[i, j] = 1 / (1 + (Dij / cutoff) ** (2 * order))

    return H

def low_pass_gaussian_filter(img, cutoff):
    H = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Dij = dij(img.shape, i, j)
            H[i, j] = np.exp(-(Dij ** 2) / (2 * (cutoff ** 2)))

    return H

def make_low_pass_filter(filter_type):
    __name__ = 'filtro passa-baixa'

    if filter_type == 'butterworth':
        return low_pass_butterworth_filter
    elif filter_type == 'gaussian':
        return low_pass_gaussian_filter
    elif filter_type == 'ideal':
        return low_pass_ideal_filter
    else:
        raise Exception('Invalid filter type')
    
pass_band_types = ['low', 'high']
filter_types = ['butterworth', 'gaussian', 'ideal']


def apply_filter(img, cutoff: int, filter_factory, filter_type: str, order=None):
    if filter_type not in filter_types:
        raise Exception('Invalid filter type')

    f = filter_factory(filter_type)

    if filter_type == 'butterworth':
        if order is None:
            order = static_order

        return f(img, cutoff, order)

    return f(img, cutoff)

def process_image(img_path: str, cutoff: int, filter_factory, filter_type: str, order=None):
    img = np.array(Image.open('./images/' + img_path).convert('L'))

    F, phase = fourier_transform(img)

    H = apply_filter(img, cutoff, filter_factory, filter_type, order)

    G = F * H

    g = inverse_fourier_transform(G, phase).astype(np.uint8)

    fig, axis = plt.subplots(2, 3)

    plt.suptitle(f'Imagem {img_path}; {filter_type} com frequência de corte {cutoff}')

    axis[0][0].imshow(img, cmap='gray')
    axis[0][0].set_title('f(x, y)')

    axis[0][1].imshow(F, cmap='gray')
    axis[0][1].set_title('F(u, v)')

    axis[0][2].imshow(H, cmap='gray')
    axis[0][2].set_title('H(u, v)')

    axis[1][0].imshow(G, cmap='gray')
    axis[1][0].set_title('F(u, v) * H(u, v)')

    axis[1][1].imshow(g, cmap='gray')
    axis[1][1].set_title('g(x, y)')

    axis[1][2].imshow(img - g, cmap='gray')
    axis[1][2].set_title('f(x, y) - g(x, y)')

    plt.show()


static_order = 1

for img_path in os.listdir('./images'):
    for filter_type in filter_types:
        process_image(img_path, 25, make_low_pass_filter, filter_type)

#Exercicio 3

def high_pass_ideal_filter(img, cutoff):
    return 1 - low_pass_ideal_filter(img, cutoff)

def high_pass_butterworth_filter(img, cutoff, order):
    H = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Dij = dij(img.shape, i, j)
            H[i, j] = 1 / (1 + (cutoff / Dij) ** (2 * order))

    return H

def high_pass_gaussian_filter(img, cutoff):
    return 1 - low_pass_gaussian_filter(img, cutoff)

def make_high_pass_filter(filter_type):
    __name__ = filter_type

    if filter_type == 'butterworth':
        return high_pass_butterworth_filter
    elif filter_type == 'gaussian':
        return high_pass_gaussian_filter
    elif filter_type == 'ideal':
        return high_pass_ideal_filter
    else:
        raise Exception('Invalid filter type')
    
for filter_type in filter_types:
    for img_path in os.listdir('./images'):
        process_image(img_path, 25, make_high_pass_filter, filter_type)

for filter_type in filter_types:
    for cutoff in [0.01, 0.05, 0.5]:
        for img_path in os.listdir('./images'):
            process_image(img_path, cutoff, make_low_pass_filter, filter_type)

#Exercicio 6

#O filtro passa-banda é um tipo de filtro que permite a passagem de frequências
# dentro de uma faixa específica. Ele é formado por dois filtros passa-baixa, 
# um com frequência de corte inferior e outro com frequência de corte superior. 
# A imagem resultante é obtida subtraindo a imagem filtrada com a frequência de 
# corte inferior da imagem filtrada com a frequência de corte superior.

def make_band_pass_filter(filter_type):
    __name__ = filter_type

    if filter_type == 'butterworth':
        return lambda img, _, __: high_pass_butterworth_filter(img, cutoff_low,
                                                               static_order) * low_pass_butterworth_filter(img,
                                                                                                           cutoff_high,
                                                                                                           static_order)
    elif filter_type == 'gaussian':
        return lambda img, _: high_pass_gaussian_filter(img, cutoff_low) * low_pass_gaussian_filter(img, cutoff_high)
    elif filter_type == 'ideal':
        return lambda img, _: high_pass_ideal_filter(img, cutoff_low) * low_pass_ideal_filter(img, cutoff_high)
    else:
        raise Exception('Invalid filter type')


cutoff_low = 0.05
cutoff_high = 0.1

for filter_type in filter_types:
    for img_path in os.listdir('./images'):
        process_image(img_path, 25, make_band_pass_filter, filter_type)

    