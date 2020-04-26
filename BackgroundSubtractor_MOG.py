import numpy as np
import cv2 as cv

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html


cap = cv.VideoCapture(0)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv.imshow('MOG',frame)
    cv.imshow('frame',fgmask)

    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv.destroyAllWindows()