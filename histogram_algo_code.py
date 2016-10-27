import cv2
import numpy as np
import math

def histogramEquilisation(img,rev = False):
	
	histogram = {}
	output = np.zeros((img.shape[0],img.shape[1]),np.uint8)

	for j in xrange(img.shape[0]):
		for k in xrange(img.shape[1]):

				if img[j,k] in histogram:
					histogram[img[j,k]]+=1
				else:
					histogram[img[j,k]] = 0


	keys = sorted(histogram,reverse = rev)

	for i in xrange(1,len(keys)):
		
		histogram[keys[i]] = (histogram[keys[i]] + histogram[keys[i-1]])

	for i in xrange(1,len(keys)):

		histogram[keys[i]] = (255 * histogram[keys[i]])/(img.shape[0]*img.shape[1])

	for j in xrange(img.shape[0]):
		for k in xrange(img.shape[1]):

				if img[j,k] in histogram:
					output[j,k] = histogram[img[j,k]]

	return output


for sample in range(1,2):

	img = cv2.imread('sample_low_contrast/'+str(sample)+'.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	b_row = np.zeros(img.shape[1],np.uint8)
	r_row = np.zeros(img.shape[1],np.uint8)
	g_row = np.zeros(img.shape[1],np.uint8)
	b = np.zeros((img.shape[0],img.shape[1]),np.uint8)
	g = np.zeros((img.shape[0],img.shape[1]),np.uint8)
	r = np.zeros((img.shape[0],img.shape[1]),np.uint8)

	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			
			b1,g1,r1 = img[i,j]

			b_row[j] = b1
			r_row[j] = r1
			g_row[j] = g1

		b[i] = b_row
		r[i] = r_row
		g[i] = g_row

	#b = histogramEquilisation(b)
	#g = histogramEquilisation(g)
	r = histogramEquilisation(r)

	equilized_image = np.zeros((img.shape[0],img.shape[1],3),np.uint8)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):

			equilized_image[i,j] = [b[i,j],g[i,j],r[i,j]]


	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	cv2.imshow('Input Image', img)

	equilized_image = cv2.cvtColor(equilized_image, cv2.COLOR_HSV2BGR)
	cv2.imshow('Equilised Image',equilized_image)

# cv2.imshow('Input Blue', b)
# cv2.imshow('Input Green', g)
# cv2.imshow('Input Red', r)


	cv2.waitKey(0)
cv2.destroyAllWindows()