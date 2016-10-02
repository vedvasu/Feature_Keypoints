import cv2
import numpy as np
import math

img = cv2.imread('cameraman.tif')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

negative = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
log_transfrom = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
intensity_slicing = np.zeros((img.shape[0],img.shape[1],3),np.uint8) 
bit_plane_1 = np.zeros((img.shape[0],img.shape[1]),np.uint8) 
bit_plane_2 = np.zeros((img.shape[0],img.shape[1]),np.uint8) 
bit_plane_3 = np.zeros((img.shape[0],img.shape[1]),np.uint8) 
bit_plane_4 = np.zeros((img.shape[0],img.shape[1]),np.uint8) 

c = 50

for i in range(0,img.shape[1]):
	for j in range(0,img.shape[0]):
		
		negative[i,j][0] = 255 - img[i,j][0]
		negative[i,j][1] = 255 - img[i,j][1]
		negative[i,j][2] = 255 - img[i,j][2]

		log_transfrom[i,j] = c*math.log(gray[i,j]+1)
		
		if gray[i,j] > 50 and gray[i,j] < 200:
			intensity_slicing[i,j] = 255

		value = bin(gray[i,j])

		bit_plane_1[i,j] = 255 if value[len(value)-1] == '1' else 0
		bit_plane_2[i,j] = 255 if value[len(value)-2] == '1' else 0
		bit_plane_3[i,j] = 255 if value[len(value)-3] == '1' else 0
		bit_plane_4[i,j] = 255 if value[len(value)-4] == '1' else 0

print 
print "-> Ngative = 255 - pixel_value"
print "-> In log ransform: S = c*log(1+r)....c = 50"
print "-> In intensity level slicing: range = [50,200]"


cv2.imshow('originalImage',img)
cv2.imshow('negative',negative)
cv2.imshow('log_transform',log_transfrom)
cv2.imshow('intensity_slicing',intensity_slicing)
cv2.imshow('bit_plane1',bit_plane_1)
cv2.imshow('bit_plane2',bit_plane_2)
cv2.imshow('bit_plane3',bit_plane_3)
cv2.imshow('bit_plane4',bit_plane_4)

cv2.waitKey(0)
cv2.destroyAllWindows()