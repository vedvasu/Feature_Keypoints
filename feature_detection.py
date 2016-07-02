
'''
* File name: feature_detection.py
* Author List: Ved Vasu Sharma
* Classes created: SIFTFeatures(), ORBFeatures(), Operatio_On_Keypoints() 
* Description: - This module aims at identifying feature from an image using inbuilt feature detectors.
               - Initially developed using SIFT and ORB feature.
* Example Call: s = SIFTFeatures()
                keyPoints = s.transform(img,None)     # None is the mask as required by detectAndCompute feature
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)                    # to view the large array in uncompressed form

#f = open('results/output_test_image1_histogram.txt','w')
#sys.stdout = f

class SIFTFeatures():
    
    '''
    * SIFT is scale invariant feature transform used to detect keypoints and its parameters
    * These parameters can be filtered depending upon
   			- its (x,y) coordinates, 
			- size of the meaningful neighbourhood, 
			- angle which specifies its orientation, 
			- response that specifies strength of keypoints etc.
    '''
    
    def __init__(self):
        self.siftDesc = cv2.SIFT()

    def transform(self,X,mask):
        Y = X.copy()
        X = cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
        kp,des = self.siftDesc.detectAndCompute(X,mask)
        img=cv2.drawKeypoints(Y,kp,flags=0)					# use flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv2.imshow('key',img)								# the orientation, and size of keypoint can be viewed with this flag
        return kp,des


class ORBFeatures():                        ## Still to be implemented
    
    def __init__(self):
        self.orb = cv2.ORB()
    
    def transform(self,X,mask):
        Y = X.copy()
        X = cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
        kp = self.orb.detect(X,mask)
        kp, des = self.orb.compute(X, kp)
        img = cv2.drawKeypoints(Y,kp, flags=0)
        cv2.imshow('key',img)
        return kp,des


class Operation_On_Keypoints():                   

    def __init__(self,kp):
        self.key_points = kp

    def showKp(self,arr):
        cv2.imshow('Key_point', self.key_points[arr])

class matchingFeatures():

    def __init__(self,img1_path,img2_path):
        
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)

    def setup(self,min_match):

        s = SIFTFeatures()
        kp1,des1 = s.transform(self.img1,None)
        kp2,des2 = s.transform(self.img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > min_match:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w,t = self.img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            #self.img2 = cv2.polylines(self.img2,[np.int32(dst)],True,255,3)

        else:
            print "Not enough matches are found - %d/%d" % (len(good),min_match)
            matchesMask = None

        raw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        cv2.imshow('img2',self.img2)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        plt.imshow(img3, 'gray'),plt.show()

#################################################### SIFT feature        
# img = cv2.imread('samples/sample (7).jpg')
# s = SIFTFeatures()
# keyPoints = s.transform(img,None)
# print keyPoints

###################################################### ORB feature
# img = cv2.imread('samples/sample (7).jpg')
# s = ORBFeatures()
# keyPoints = s.transform(img,None)
# print keyPoints

##################################################### Matching Key_points
s = matchingFeatures(img1_path = 'samples/sample (1).jpg',img2_path = 'samples/sample (8).jpg') 
s.setup(min_match = 10)

# sd = Operation_On_Keypoints(keyPoints)
# sd.showKp(1)

cv2.waitKey(0)
cv2.destroyAllWindows()