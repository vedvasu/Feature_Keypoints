
'''
* File name: complete_detection.py
* Author List: Ved Vasu Sharma
* Classes created: CascadedDetector(), setupDetector()
* Description: - This module aims at detecting parts of the body using inbuilt haarcascade files in openCV.
               - Initially developed for face, nose and ears and can be extended by adding more keys.
* Example Call: s = setupDetectors('file_name.jpg',detectorKey = 'face')     # key can be face, nose, eyes
                cropped_parts = s.detect()  
'''

import cv2
import numpy as np

class CascadedDetector():

    """
    Uses the OpenCV cascades to perform the detection. Returns the Regions of Interest, where
    the detector assumes a face. You probably have to play around with the scaleFactor, 
    minNeighbors and minSize parameters to get good results for your use case. From my 
    personal experience, all I can say is: there's no parameter combination which *just 
    works*. 
    """
    
    def __init__(self, cascade_fn, scaleFactor=1.2, minNeighbors=5, minSize=(30,30)):
    
        self.cascade = cv2.CascadeClassifier(cascade_fn)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize
    
    
    def detect(self, src):
    
        if np.ndim(src) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = cv2.equalizeHist(src)
        rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
     
        if len(rects) == 0:
            return np.ndarray((0,))
        rects[:,2:] += rects[:,:2]
     
        return rects


class setupDetectors():
    
    '''
    * Input variables: path = souce path of the test_case
                       detectorKey = type of detection (can be face, nose, eyes)
    * Output: returns array containing all the parts detected as per Key
              displays the output detection on the image              
    '''
    
    def __init__(self,path,detectorKey):
        
        self.img = np.array(cv2.imread(path), dtype=np.uint8)
        self.imgOut = self.img.copy()
        self.key = detectorKey 
        self.crop = []

    def detect(self):

        if self.key == 'face':
            face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
            faces = face_cascade.detectMultiScale(self.img, 1.3, 5)
            for (x,y,w,h) in faces:
                face = self.imgOut[y:y+h, x:x+w]
                self.crop.append(face)
                #cv2.imwrite('face['+str(j)+'].jpg',face)

        if self.key == 'nose':
            nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')
            noses = nose_cascade.detectMultiScale(self.img, 1.3, 5)
            for (x1,y1,w1,h1) in noses:
                nose = self.imgOut[y1:y1+h1,x1:x1+w1]
                self.crop.append(nose)
                #cv2.imwrite('nose['+str(j)+'].jpg',nose)

        if self.key == 'eyes':
            eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(self.img, 1.3, 5)
            for (x2,y2,w2,h2) in eyes:
                eye = self.imgOut[y2:y2+h2,x2:x2+w2]
                self.crop.append(eye)
                #cv2.imwrite('eye['+str(j)+'].jpg',eye)i
        
        if self.key == 'mouth':
            mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
            mouth = mouth_cascade.detectMultiScale(self.img, 1.3, 5)
            for (x3,y3,w3,h3) in mouth:
                mouth_ = self.imgOut[y3:y3+h3,x3:x3+w3]
                self.crop.append(mouth_)

        #cv2.imshow('Output',self.imgOut)
        
        return self.crop 


#### Main Program ########

#s = setupDetectors('samples/sample (3).jpg',detectorKey = 'face')     # key can be face, nose, eyes
#cropped_parts = s.detect()

#if cropped_parts != []:
#    cv2.imshow('detection',cropped_parts[0])


#cv2.waitKey(0)
#cv2.destroyAllWindows()
