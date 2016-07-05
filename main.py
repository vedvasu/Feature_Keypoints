import cv2
import numpy as np
import complete_detection as roi
import feature_detection as feature

# Input Image
path = 'samples/sample (7).jpg'
#img = cv2.imread('samples/sample (7).jpg')

# Step 1 : Extraction of face from the image (Region of Interest) 
s = roi.setupDetectors(path,detectorKey = 'face')     # key can be face, nose, eyes
cropped_parts = s.detect()
face = cropped_parts[0]

# Step 2 : Feature descriptor (Using Sift Feature)
s = feature.SIFTFeatures()
keyPoints,descriptor = s.transform(face,None)
print len(keyPoints)

# Step 3 : Keypoints Filtering 					#Still to be implemented



cv2.waitKey(0)
