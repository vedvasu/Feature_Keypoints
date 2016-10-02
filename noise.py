import cv2
import numpy
import random

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

img = cv2.imread('cameraman.tif',0)

for i in range(500):
	img[random.randint(0,255),random.randint(0,255)] = 0
	img[random.randint(0,255),random.randint(0,255)] = 255

mean_filter = cv2.blur(img,(3,3))
median_filter = cv2.medianBlur(img,3)

min_filter = cv2.erode(img,kernel,iterations = 1)
max_filter = cv2.dilate(img, kernel, iterations = 1)

opening = cv2.dilate(min_filter, kernel, iterations = 1)
closing = cv2.erode(max_filter, kernel, iterations = 1)


cv2.imshow('originalImage',img)
cv2.imshow('mean_filter',mean_filter)
cv2.imshow('median_filter',median_filter)
cv2.imshow('max_filter',max_filter)
cv2.imshow('min_filter',min_filter)
cv2.imshow('opening',opening)
cv2.imshow('closing',closing)

cv2.waitKey(0)
cv2.destroyAllWindows()