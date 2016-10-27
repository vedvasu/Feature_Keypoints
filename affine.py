import cv2
import numpy as np
import math

img = cv2.imread('sample_low_contrast/2.jpg')

########################## Inbuilt functions ###############################################

# M = np.float32([[1,0,15],[0,1,10]])
# translation = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

# M1 = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),60,1)
# rotation = cv2.warpAffine(img,M1,(img.shape[1],img.shape[0]))

resized = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

############################################################################################

# translation 
translation = np.zeros((img.shape[0],img.shape[1],3),np.uint8)

for i in range(15,img.shape[0]):
	for j in range(10,img.shape[1]):

		translation[i,j] = img[i,j]

# rotation
rotation = np.zeros((img.shape[0],img.shape[1],3),np.uint8)

cx = img.shape[0]/2
cy = img.shape[1]/2
angle = 60
alpha = math.cos(math.radians(angle))
beta = math.sin(math.radians(angle))

M1 = np.array([[alpha,beta,((1-alpha)*cx - beta*cy)],[-beta,alpha,(beta*cx + (1-alpha)*cy)]],np.float32)

print 
print '-> Rotation'
print 'aplha =',alpha,'beta =',beta
print 'Transformation matrix ='
print M1

for i in range(0,img.shape[0]):
	for j in range(0,img.shape[1]):

		if (M1[0][0]*i+M1[0][1]*j+M1[0][2]) < img.shape[0] and (M1[0][0]*i+M1[0][1]*j+M1[0][2]) > 0 and (M1[1][0]*i+M1[1][1]*j+M1[1][2]) < img.shape[1] and (M1[1][0]*i+M1[1][1]*j+M1[1][2]) > 0:
			rotation[i,j] = img[int(M1[0][0]*i+M1[0][1]*j+M1[0][2]),int(M1[1][0]*i+M1[1][1]*j+M1[1][2])]

m = 0.5
wrapaffine = np.zeros((img.shape[0],(int(m*img.shape[1])+img.shape[1]),3),np.uint8)

initial_value = 0
for i in range(0,img.shape[0]):
	for j in range(0,img.shape[1]):

		wrapaffine[i,int(j+(m*initial_value))] = img[i,j]
	initial_value+=1

cv2.imshow('originalImage',img)
cv2.imshow('translation',translation)
cv2.imshow('rotation',rotation)
cv2.imshow('scaling',resized)
cv2.imshow('wrapaffine',wrapaffine)

cv2.waitKey(0)
cv2.destroyAllWindows()