import cv2
import numpy

img = cv2.imread('sample_low_contrast/2.jpg')

blur = cv2.blur(img,(5,5))

deblur = cv2.fastNlMeansDenoisingColored(blur,None,5,5,21,7)

cv2.imshow('originalImage',img)
cv2.imshow('blur',blur)
cv2.imshow('deblur',deblur)

cv2.waitKey(0)
cv2.destroyAllWindows()