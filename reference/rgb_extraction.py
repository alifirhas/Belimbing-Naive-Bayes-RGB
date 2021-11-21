import cv2
import numpy

# ! Format pembacaan warna CV2 adalah BGR [Blue Gree Red]
# Hati hati dengan format file directory
img = cv2.imread('/home/alifirhas/0_Work/0_Kuliah/Semester 5/0_Project/0_Belimbing/Belimbing-Naive-Bayes-RGB/images/hamster.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # ! Konfersi BGR2RGB

avg_color_per_row = numpy.average(img_rgb, axis=0)          # Average nilai yang ada di row
avg_color = numpy.average(avg_color_per_row, axis=0)        # Average nilai yang ada di column
print(avg_color)

# cv2.imshow('ImageWindow', img)  # Tampil kan gambar
# cv2.waitKey()                   # ! Harus ada waitkey