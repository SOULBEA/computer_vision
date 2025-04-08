from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils

# construct the argument parser and prase the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
                help = "path to the image to be scanned")
args = vars(ap.parse_args())
