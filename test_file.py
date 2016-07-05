import cv2
import numpy as np

# mouse callback function
def pixel_value(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:    # USE cv2.EVENT_MOUSEMOVE for continuous value display
        print img[x,y]

img = cv2.imread('samples/sample (1).jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',pixel_value)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(0):
        break
cv2.destroyAllWindows()