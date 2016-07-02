
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
            detector = CascadedDetector(cascade_fn="haarcascades\haarcascade_frontalface_alt2.xml")      
            for i,r in enumerate(detector.detect(self.img)):
                x0,y0,x1,y1 = r
                cv2.rectangle(self.imgOut, (x0,y0),(x1,y1),(0,255,0),1)
                
                face = self.imgOut[y0:y1,x0:x1]
                self.crop.append(face)
                #cv2.imwrite('face['+str(j)+'].jpg',face)

        if self.key == 'nose':
            noseDetector = CascadedDetector(scaleFactor=1.1,minNeighbors=5, minSize=(20,20), cascade_fn="haarcascades\haarcascade_mcs_nose.xml")            

            for i,r in enumerate(noseDetector.detect(self.img)):
                fx0,fy0,fx1,fy1 = r
                cv2.rectangle(self.imgOut, (fx0,fy0),(fx1,fy1),(255,0,0),1)
                
                nose = self.imgOut[fy0:fy1,fx0:fx1]
                self.crop.append(nose)
                #cv2.imwrite('nose['+str(j)+'].jpg',nose)

        if self.key == 'eyes':
            eyesDetector = CascadedDetector(scaleFactor=1.1,minNeighbors=5, minSize=(20,20), cascade_fn="haarcascades\haarcascade_eye.xml")


            for j,r1 in enumerate(eyesDetector.detect(self.img)):
                ex0,ey0,ex1,ey1 = r1
                cv2.rectangle(self.imgOut, (ex0,ey0),(ex1,ey1),(0,255,0),1)
                
                eye = self.imgOut[ey0:ey1,ex0:ex1]
                self.crop.append(eye)
                #cv2.imwrite('eye['+str(j)+'].jpg',eye)

        #cv2.imshow('Output',self.imgOut)
        
        return self.crop 


#### Main Program ########

# s = setupDetectors('samples/sample (1).jpg',detectorKey = 'face')     # key can be face, nose, eyes
# cropped_parts = s.detect()

# if cropped_parts != []:
#     cv2.imshow('detection',cropped_parts[0])


cv2.waitKey(0)