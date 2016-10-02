import cv2
import numpy as np
import math

img = cv2.imread('cameraman.tif',0)
img1 = cv2.imread('lion.tif')
img2 = cv2.imread('fruits.tif')

equ = cv2.equalizeHist(img)

cv2.imshow('originalImage',img)
cv2.imshow('Equilised Image',equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('originalImage',img)
cv2.imshow('Comparision 1',img1)
cv2.imshow('Comparision 2',img2)

print
print "-> Histogram Comparision"
print
h = cv2.calcHist([img],[0],None,[256],[0,256])
h1 = cv2.calcHist([img1],[0],None,[256],[0,256])
h2 = cv2.calcHist([img2],[0],None,[256],[0,256])
print 
print "-> Correlation based histogram matching"
print cv2.compareHist(h, h, cv2.cv.CV_COMP_CORREL)
print cv2.compareHist(h, h1, cv2.cv.CV_COMP_CORREL)
print cv2.compareHist(h, h2, cv2.cv.CV_COMP_CORREL)

print "-> Intersection based histogram matching"
print cv2.compareHist(h, h, cv2.cv.CV_COMP_INTERSECT)
print cv2.compareHist(h, h1, cv2.cv.CV_COMP_INTERSECT)
print cv2.compareHist(h, h2, cv2.cv.CV_COMP_INTERSECT)

cv2.waitKey(0)
cv2.destroyAllWindows()