import numpy as np 
import cv2 
import math
import scipy 
from scipy.signal import medfilt

def hard_threshold (img, epsilon):
    image = img/255
    for i in range (0, image.shape[0]):     #rows
        for j in range (0, image.shape[1]):     #columns
            if image[i][j] >= epsilon:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image*255

def soft_threshold (img, epsilon, phi):
    image = img/255
    for i in range (0, image.shape[0]):     #rows
        for j in range (0, image.shape[1]):     #columns
            if image[i][j] >= epsilon:
                image[i][j] = 1
            else:
                image[i][j] = 1 + np.tanh(phi * (image[i][j] - epsilon))
    return image*255

in_img = cv2.imread('XDoG.jpeg',0)
hard_img = hard_threshold(in_img, 0.5)
cv2.imwrite("Hard_Threshold_shark.png", hard_img)

soft_img = soft_threshold(in_img, 0.5, 6)
cv2.imwrite("Soft_Threshold_shark.png", soft_img)