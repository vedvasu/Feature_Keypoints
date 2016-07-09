
'''
* File name: feature_detection.py
* Author List: Ved Vasu Sharma (logics copied from openCV dcumentation)
* Classes created: SIFTFeatures(), ORBFeatures(), Operatio_On_Keypoints() ,matchingFeatures()
* Description: - This module aims at identifying feature from an image using inbuilt feature detectors.
               - Initially developed using SIFT and ORB feature.
* Example Call: s = SIFTFeatures()
                keyPoints = s.transform(img,None)     # None is the mask as required by detectAndCompute feature
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)                    # to view the large array in uncompressed form

#f = open('results/output_test_image1_histogram.txt','w')     # uncomment to save the output as .txt (output not printed on console)
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
        img=cv2.drawKeypoints(Y,kp,flags=0)                 # use flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv2.imshow('key',img)                               # the orientation, and size of keypoint can be viewed with this flag
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


class operationsKeypoints():                   

    '''
    * Purpose of this class is to write codes for the operations not working in openCV
    * New basic implementations on the keypoints
    '''

    def drawMatches(self,img1, kp1, img2, kp2, matches):

        '''
        * Alternate implementation for cv2.drawMatches function in openCV
        * The keypoints which are matched intwo images are drawn(connected by a line) on the image formed by joining both images.
        * The function returns out put image with drawn keypoints
        '''
              
        ## New black image having max row of the both samples......that means images will be stacked horizontally
        out = np.zeros((max([img1.shape[0],img2.shape[0]]),img1.shape[1]+img2.shape[1],3), dtype='uint8')

        out[:img1.shape[0],:img1.shape[1]] = np.dstack([img1])      # sample pixels overwritten on new image
        out[:img2.shape[0],img1.shape[1]:] = np.dstack([img2])

        for m in matches:

            img1_idx = m.queryIdx
            img2_idx = m.trainIdx

            (x1,y1) = kp1[img1_idx].pt                  # gives the location of the point (x,y) of the keypoint
            (x2,y2) = kp2[img2_idx].pt

            ## Figures are drawn
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 255), 1)   
            cv2.circle(out, (int(x2)+img1.shape[1],int(y2)), 4, (255, 0, 255), 1)

            cv2.line(out, (int(x1),int(y1)), (int(x2)+img1.shape[1],int(y2)), (255, 0, 0), 1)

        return out
    
    def query2pointsConversion(self,kp):

        points = []
        for i in xrange(len(kp)):

            (x,y) = kp[i].pt                  # gives the location of the point (x,y) of the keypoint
            points.append([int(x),int(y)])
        
        return points


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
            
            out = operationsKeypoints().drawMatches(self.img1,kp1,self.img2,kp2,good)
            cv2.imshow('out',out)

        else:
            print "Not enough matches are found - %d/%d" % (len(good),min_match)

        return good


#################################################### SIFT feature        
# img = cv2.imread('samples/sample (7).jpg')
# s = SIFTFeatures()
# keyPoints, des = s.transform(img,None)
# print keyPoints

###################################################### ORB feature
# img = cv2.imread('samples/sample (7).jpg')
# s = ORBFeatures()
# keyPoints, des = s.transform(img,None)
# print keyPoints

##################################################### Matching Key_points
# s = matchingFeatures(img1_path = 'samples/sample (1).jpg',img2_path = 'samples/sample (8).jpg') 
# matches = s.setup(min_match = 10)

#################################################### Operations
# points = operationsKeypoints().query2pointsConversion(keyPoints)

cv2.waitKey(0)
cv2.destroyAllWindows()