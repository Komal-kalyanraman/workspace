import numpy as np
import cv2 as cv

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html

cap = cv.VideoCapture(0)

#kernel = cv.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
kernel = np.ones((3,3),np.uint8)

fgbg = cv.bgsegm.createBackgroundSubtractorGMG()


while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    #fgmask = cv.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    erosion = cv.erode(fgmask,kernel,iterations = 1)
    dilation = cv.dilate(erosion,kernel,iterations = 1)

    cv.imshow('original',frame)

    cv.imshow('GMG',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()