import cv2
import numpy as np
import math

img = cv2.imread('sample_low_contrast/1.jpg',0)

histogram = {}
output = np.zeros((img.shape[0],img.shape[1]),np.uint8)
for j in xrange(img.shape[0]):
	for k in xrange(img.shape[1]):

			if img[j,k] in histogram:
				histogram[img[j,k]]+=1
			else:
				histogram[img[j,k]] = 0


keys = sorted(histogram)

for i in xrange(1,len(keys)):
	
	histogram[keys[i]] = (histogram[keys[i]] + histogram[keys[i-1]])

for i in xrange(1,len(keys)):

	histogram[keys[i]] = (255 * histogram[keys[i]])/(img.shape[0]*img.shape[1])

print histogram

for j in xrange(img.shape[0]):
	for k in xrange(img.shape[1]):

			if img[j,k] in histogram:
				output[j,k] = histogram[img[j,k]]


cv2.imshow('Input Image', img)
cv2.imshow('Output Image', output)

cv2.waitKey(0)
cv2.destroyAllWindows()