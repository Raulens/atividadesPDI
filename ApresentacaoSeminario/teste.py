import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def detectar_bordas(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    #Canny
    bordas = cv2.Canny(imagem_cinza, 30, 150)
    
    return bordas

def verificar_spoofing(imagem_real, imagem_spoof):
    bordas_real = detectar_bordas(imagem_real)
    bordas_spoof = detectar_bordas(imagem_spoof)

    plt1 = plt.subplot(1,2,1)
    plt2 = plt.subplot(1,2,2)
    plt1.title.set_text('Real image')
    plt2.title.set_text('Spoof image')

    plt1.imshow(bordas_real)
    plt2.imshow(bordas_spoof)
    plt.show()
    
    diferenca = cv2.absdiff(bordas_real, bordas_spoof)
    
    limiar = 30
    
    pixels_diferentes = np.sum(diferenca > limiar)
    
    limite_pixels_diferentes = 10000
    
    if pixels_diferentes > limite_pixels_diferentes:
        return "Impressão digital falsa detectada (spoofing)."
    else:
        return "Impressão digital autêntica."

imagem_real = cv2.imread("image1.png")

imagem_spoof = cv2.imread("image1.png")

resultado = verificar_spoofing(imagem_real, imagem_spoof)

print(resultado)