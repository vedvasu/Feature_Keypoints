import cv2
import numpy as np

img = cv2.imread('cameraman.tif')
print '-> Quantisation'
print 'shape = ',img.shape

gray_lev8 = np.zeros((img.shape[0],img.shape[1]),np.uint8)
gray_lev2 = np.zeros((img.shape[0],img.shape[1]),np.uint8)
gray_lev4 = np.zeros((img.shape[0],img.shape[1]),np.uint8)

for i in range(img.shape[1]):
	for j in range(img.shape[0]):
		r,g,b = img[i,j]
		value = (0.21*r)+(0.72*g)+(0.07*b)
		gray_lev8[i,j] = value

		gray_lev2[i,j] = 255 if value >= 127 else 0
		gray_lev4[i,j] = value/16

		gray_lev4[i,j] = (gray_lev4[i,j] << 4);


print 'Original Image',img
print '8 bit Image',gray_lev8
print '2 bit Image',gray_lev2
print '4 bit Image',gray_lev4

cv2.imshow('originalImage',img)
cv2.imshow('gray_lev8',gray_lev8)
cv2.imshow('gray_lev2',gray_lev2)
cv2.imshow('gray_lev4',gray_lev4)

cv2.waitKey(0)
cv2.destroyAllWindows()