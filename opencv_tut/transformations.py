import cv2 as cv
import numpy as np

img = cv.imread('baki_pf.jpg')
cv.imshow('baki', img)

def translate(img, x, y):
    transMat = np.float32([[1, 0, x],[0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

#rotation
def rotate(img, angle, rotPoint = None):
    (height, widht) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (widht//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (widht, height)

    return cv.warpAffine(img, rotMat, dimension)

rotated = rotate(img, 45)
cv.imshow('rotated', rotated)

translated = translate(img, 100, 100)
cv.imshow('translated', translated)

cv.waitKey(0)
