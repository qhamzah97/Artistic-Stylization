import numpy as np 
import cv2 
import math
import scipy 
from scipy.signal import medfilt

def three_tone (img, epsilon1, epsilon2, epsilon3, phi):
    image = img/255
    for i in range (0, image.shape[0]):     #rows
        for j in range (0, image.shape[1]):     #columns
            if image[i][j] >= 0 and image[i][j] < epsilon1:     # 0 to 0.3
                image[i][j] = 0.25*np.tanh(phi*(image[i][j] - 0)) + 0.5 + 0.25*np.tanh(phi*(image[i][j]-epsilon1))
            elif image[i][j] >= epsilon1 and image[i][j] < epsilon2:        # 0.3 to 0.4
                image[i][j] = 1
            elif image[i][j] >= epsilon2 and image[i][j] < epsilon3:        # 0.4 to 0.6
                image[i][j] = 0.25*np.tanh(phi*(image[i][j] - epsilon2)) + 0.5 + 0.25*np.tanh(phi*(image[i][j]-epsilon3))
            elif image[i][j] >= epsilon3 and image[i][j] <= 1:      # 0.6 to 1
                image[i][j] = 1
    return image*255

in_img = cv2.imread('XDoG.jpeg',0)
three_tone_img = three_tone(in_img, 0.3, 0.4, 0.6, 6)
cv2.imwrite("Three_Tone_shark.png", three_tone_img)
