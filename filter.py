import cv2
import numpy

k = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

img = cv2.imread('sample_low_contrast/2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print
print "-> 3x3 averaging filter"
blur = cv2.blur(img,(3,3))

laplacian = cv2.Laplacian(blur,cv2.CV_64F)
sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)

cv2.imshow('originalImage',img)
cv2.imshow('blur',blur)
cv2.imshow('laplacian',laplacian)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()

print "-> 5x5 averaging filter"
blur = cv2.blur(img,(5,5))

laplacian = cv2.Laplacian(blur,cv2.CV_64F)
sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)

cv2.imshow('originalImage',img)
cv2.imshow('blur',blur)
cv2.imshow('laplacian',laplacian)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()