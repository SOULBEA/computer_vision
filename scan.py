from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap = add_argument("-i", "--image", required = True,
                  help = "path to the image to be scanned")
args = vars(ap.parse_args())
ap.add_argumenmt
