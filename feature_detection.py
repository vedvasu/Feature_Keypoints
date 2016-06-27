import cv2
import numpy as np

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
        return des


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
        return des


class Operation_On_Keypoints:                   

    def __init__(self,kp):
        self.key_points = kp

    def showKp(self,arr):
        cv2.imshow('Key_point', self.key_points[arr])

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


# sd = Operation_On_Keypoints(keyPoints)
# sd.showKp(1)

cv2.waitKey(0)
cv2.destroyAllWindows()