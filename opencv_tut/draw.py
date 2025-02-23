import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
#cv.imshow('blank', blank)

# paint the image a certain color
blank[200:300, 300:400] = 0,255,0
cv.imshow('green', blank)

cv.waitKey(0)

