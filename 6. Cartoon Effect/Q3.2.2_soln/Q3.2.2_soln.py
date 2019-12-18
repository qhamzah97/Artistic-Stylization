import numpy as np
import cv2
import scipy
import math
from scipy import signal
from scipy.ndimage import convolve

img = cv2.imread("orca.png",0)
pi =math.pi
e = math.e
T = float(input("Give 'T' a number between 0 or 1 input: \n"))
while T<0.0 or T>1.0:
    print("invalid input choose another value for B")
    T = float(input("Give 'T' either 0 or 1 input:"))

r = int(input("Give a radius for the size of the blurring kernel: \n"))
while r<0:
    print("invalid input choose another value for r")
    r = int(input("Give a radius value for the size of the blurring kernel : \n"))
    
# Creating a Guassian filter
def G_kernel(r):
    r = int(r)
    diameter = r*2+1
    sigma = 0.9
    # creating a zero array that has rows and columns the size of the diamter
    kernel = np.zeros((int(diameter), int(diameter)))
    #creating the Guassian Filters
    for x in range(0,int(diameter)):
        for y in range(0,int(diameter)):
            kernel[x,y] = (1/(2*pi*(sigma**2)))*(e**(-1*(((x-r)**2 + (y-r)**2)/(2*(sigma**2)))))        
    #print(kernel)

    #Sobel filter 
    So = np.array([[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]])
    #Prewit filter
    Pr = np.array([[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]])
    #use the filter that is prefered 
    return So

# Thresholding
def image_thresholding(img, T):
    img = np.array(img)
    #print(img)
    T_img = img/255
    #print(T_img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(T_img[x,y]<T):
                T_img[x,y] =0  # if less than T = 0
            img[x,y] = 1   # if greater than or equal to t  =1
    cv2.imshow('image', T_img)
    cv2.waitKey(0)
   
    return(T_img)
#Threshold_img = image_thresholding(img,T)
#Thresheld_orca = cv2.imwrite("T_orca.png", Threshold_img)

# Calculating Gaussian Magnitude
def GM(img, r):
    kernel = G_kernel(r)
    kernel_Tran = kernel.transpose()
    # test using convolution built in function 
    output = signal.convolve2d(img,kernel, boundary = 'symm', mode ='same')
    output_Tran = signal.convolve2d(img,kernel_Tran, boundary = 'symm', mode ='same')
    GM_MAG = output #np.zeros_like(kernel, dtype = float) # convolution output into zero array for GM    
    rows, columns = GM_MAG.shape[:2]
    for i in range(0, rows):
        for j in range (0, columns):
            GM_MAG[i][j] = math.sqrt(math.pow(output[i][j],2) + math.pow(output_Tran[i][j],2))
    print(GM_MAG)
    return GM_MAG

# Finding Prominent Edges
def P_Edges(img, r):
    
    Gaussian_MAG = GM(img, r)
    P_Edges = image_thresholding(Gaussian_MAG,T)


    return P_Edges

output = P_Edges(img, r)
cv2.imwrite('Prominent Edges.png', output)