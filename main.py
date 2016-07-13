
import cv2
import numpy as np

import complete_detection as roi

import feature_detection as feature
''' '''
# Input Image
path = 'samples/sample (2).jpg'
img = cv2.imread(path)

# Step 1 : Extraction of face from the image (Region of Interest) 
s = roi.setupDetectors(path,detectorKey = 'face')     # key can be face, nose, eyes
cropped_parts = s.detect()
if cropped_parts != []:
	face = cropped_parts[0]
else:
	face = img
faceCopy = face.copy()


# Step 2 : Feature descriptor (Using Sift Feature)
s = feature.SIFTFeatures()
keyPoints,descriptor = s.transform(face,None)
##print len(keyPoints)


# Step 3 : Keypoints Filtering 					#Still to be implemented
points = feature.operationsKeypoints().query2pointsConversion(keyPoints)

for i in xrange(len(points)):

	b,g,r = face[points[i][1],points[i][0]]
	if b < 100 and g <100 and r <100:# or r > 200 and b<200:
		cv2.circle(faceCopy,(points[i][0],points[i][1]),4,(0,0,255),1)

cv2.imshow('face',faceCopy)

cv2.waitKey(0)

cv2.destroyAllWindows()
