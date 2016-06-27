import cv2
import numpy as np
#import complete_detection

test_number = 1
number_of_images = 7
np.set_printoptions(threshold=np.nan)                                   # to view the large array in uncompressed form

#f = open('results/output_test_image1_histogram.txt','w')
#sys.stdout = f

class SIFTFeatures:
    
    def __init__(self):
        self.siftDesc = cv2.SIFT()
    
    def transform(self,X):
        Y = X.copy()
        X = cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
        kp,des = self.siftDesc.detectAndCompute(X,None)
        img=cv2.drawKeypoints(Y,kp,flags=0)
        cv2.imshow('key',img)
        return des


class ORBFeatures:
    
    def __init__(self):
        self.orb = cv2.ORB()
    
    def transform(self,X):
        Y = X.copy()
        X = cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
        kp = self.orb.detect(X,None)
        kp, des = self.orb.compute(X, kp)
        img = cv2.drawKeypoints(Y,kp, flags=0)
        cv2.imshow('key',img)
        return des


class Operation_On_Keypoints:

    def __init__(self,kp):
        self.key_points = kp

    def showKp(self,arr):
        cv2.imshow('Key_point', self.key_points[arr])

        
# img = cv2.imread('samples/sample (1).jpg')
# s = SIFTFeatures()
# keyPoints = s.transform(img)
# print keyPoints


img = cv2.imread('samples/sample (7).jpg')
s = ORBFeatures()
keyPoints = s.transform(img)
print keyPoints


# sd = Operation_On_Keypoints(keyPoints)
# sd.showKp(1)

cv2.waitKey(0)
cv2.destroyAllWindows()