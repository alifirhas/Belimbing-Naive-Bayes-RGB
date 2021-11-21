"""
    Menampikan gambar asli dan gambar grayscale
    * OpenCV membaca gambar dengan format BRG
    * Jadi jika ingin memproses image yang dibukan dengan OpenCV,
    * maka image harus dikonversi ke format RGB
"""

import cv2
import numpy as np
from numpy import array, arange, uint8 
from matplotlib import pyplot as plt


img = cv2.imread('/home/alifirhas/0_Work/0_Kuliah/Semester 5/0_Project/0_Belimbing/Belimbing-Naive-Bayes-RGB/images/hamster.png', cv2.IMREAD_COLOR)
# ! Jika tidak dikovernsi maka gambar akan memiliki format warna yang salah
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # ! Konfersi BRG2RGB

bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

images = []
images.append(img)
images.append(bw_img)

titles = ['Original Image','BW Image']

for i in range(len(images)):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()