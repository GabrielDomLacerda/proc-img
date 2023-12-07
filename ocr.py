# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

# %%
def processar_imagem(image, threshold=150, gaussian_size=3, gaussian_sigma=3, median_size=1, invert=False, dilate_size=3, dilate_it=1, erode_size=3, erode_it=1):
    proc_img = image.copy()
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    
    thresh_type = cv2.THRESH_BINARY
    proc_img = cv2.threshold(proc_img, threshold, 255, thresh_type)[1]

    if invert:
        proc_img = 255 - proc_img

    proc_img = cv2.GaussianBlur(proc_img, (gaussian_size, gaussian_size), gaussian_sigma)
    proc_img = cv2.medianBlur(proc_img, median_size)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_DILATE, (dilate_size, dilate_size))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ERODE, (erode_size, erode_size))

    proc_img = cv2.dilate(proc_img, kernel_dilate, iterations=dilate_it)
    proc_img = cv2.erode(proc_img, kernel_erode, iterations=erode_it)

    return proc_img

# %%
def extrair_texto_imagem(image):
    custom_config = r'--oem 3 --psm 6'
    texto = pytesseract.image_to_string(image, config=custom_config)
    return texto.replace('  ', ' ').replace('\n\n', '\n')

# %%
def processamento_final(file: str, threshold=150, gaussian_size=3, gaussian_sigma=3, median_size=1, invert=False, dilate_size=3, dilate_it=1, erode_size=3, erode_it=1):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    _, axis = plt.subplots(1, 2)
    axis[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = processar_imagem(img, threshold, gaussian_size, gaussian_sigma, median_size, invert, dilate_size, dilate_it, erode_size, erode_it)

    axis[1].imshow(img, cmap='gray')
    return extrair_texto_imagem(img)

# %%
print(processamento_final('imgs/texto.png'))

# %%
print(processamento_final('imgs/bh.png', threshold=200))

# %%
texto = processamento_final('imgs/senhor_dos_aneis.png', threshold=100, gaussian_sigma=1, gaussian_size=5, median_size=1, erode_size=3, dilate_size=1)
print('\n'.join(list(filter(lambda x: len(x) > 3, texto.split('\n')))[-5:]))

# %%
texto = processamento_final('imgs/fake_png_crianca.png', threshold=120, erode_size=5, erode_it=3, dilate_size=2, dilate_it=3)
print(texto)


