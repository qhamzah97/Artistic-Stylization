import numpy as np
import cv2
import scipy
import math
from scipy import stats

N = 16
R = 3
G = 10

def oilify(img, N, R, gamma):
    image = img/255
    H = np.zeros((N,1))
    acc = np.zeros((N,1))
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            for i in range (x-R, x+R):
                for j in range (y-R, y+R):
                    index = math.floor(image[x][y] * (N-1))
                    H[index] = H[index] + 1
                    acc[index] = acc[index] + img[i][j]
            hMAX = stats.mode(H)
            A = 0
            B = 0
            for z in range (0, N-1):
                w = (H[z]/hMAX) ** gamma
                B = B + w
                A = A + w*(acc[z]/H[z])
            img[x][y] = A/B
    return img


in_img = cv2.imread("digital_orca_blurred.png",0)
out_img = oilify(in_img, N, R, G)
