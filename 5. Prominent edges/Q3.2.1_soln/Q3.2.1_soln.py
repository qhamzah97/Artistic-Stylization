import numpy as np 
import cv2 
import math
import scipy
from scipy.signal import medfilt

def smoothing(img, r, count):
    rows, columns = img.shape
    image = np.zeros((rows, columns))
    image = scipy.signal.medfilt2d(img, r)
    count = count - 1
    if count != 0:
        r = r + 2
        smoothing(image, r, count)
    return image

in_img = cv2.imread('digital_orca_blurred.png',0)
smooth_img = smoothing(in_img, 3, 5)
cv2.imwrite("Smooth.png", smooth_img)