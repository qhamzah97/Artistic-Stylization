import numpy as np 
import cv2 
import math
import scipy 
from scipy.ndimage import convolve
from scipy import signal

#image = cv2.imread('digital_orca_blurred.png',cv2.IMREAD_GRAYSCALE)
#image2 = image.copy()

#def gaussianblur(img,r,sig, m):

#    #blurring
#    rangevar = r*2+1
#    blurKernel = np.zeros((rangevar,rangevar))
#    for i in range(0, rangevar):
#     for j in range(0, rangevar):
#        dia = float(((i - r) ** 2) + ((j - r) ** 2))
#        yeet = float(1/(2*math.pi*(sig**2)))
#        blurKernel[i, j] = yeet*(math.exp((-1 * dia) / (2 * (sig ** 2))))  # *math.pow(k,2)
#    print(blurKernel)

#    #convolution
#    value = 0
#    newimg = np.empty([img.shape[0], img.shape[1]], dtype=float)
#    for x in range(img.shape[0]):
#        for y in range(img.shape[1]):
#            if x < r or y < r or (x >= (img.shape[0]-r)) or (y >= (img.shape[1]-r)):
#                newimg[x][y] = 0
#                pass
#            else:
#                for i in range(-1*r, r+1):
#                    for j in range(-1*r, r+1):
#                        value = value + (img[x+i, y+j]*blurKernel[r+i, r+j])
#                value = value * m
#                newimg[x][y] = value
#                value = 0
#    return newimg

##XDoG
#def XDoG(img, m):
#    blur = gaussianblur(img, 3, 0.9, m + 1)
#    blur2 = gaussianblur(img, 3, 1.2*0.9, m)

#    imgout = np.empty([blur.shape[0], blur.shape[1]])
#    for i in range(0, blur.shape[0]):
#        for j in range(0, blur.shape[1]):
#            imgout[i][j] = blur[i][j] - blur2[i][j]
#    return imgout

#final = XDoG(image, 100)
#cv2.imwrite("XDoG.jpg", final)

r = int(input("Give a radius for the size of the blurring kernel: \n"))
print(r)
while r<0:
    print("invalid input choose another value for r")
    r = int(input("Give a radius value for the size of the blurring kernel : \n"))
    print(r)
sigma = float(input("give a sigma value for the Guassian Filter \n"))
img = cv2.imread('orca.png',0)

def eXtended_DoG(img, r , sigma):
    img = (np.array(img)).astype(float)
    r = int(r)
    diameter = r*2+1
    diameter = float(diameter)
    k = 1.2
    p = 100
    # creating a zero array that has rows and columns the size of the diamter
    kernel1 = np.zeros((int(diameter), int(diameter)))
    kernel2 = np.zeros((int(diameter), int(diameter)))
    #creating the Guassian Filters
    for x in range(0,int(diameter)):
        for y in range(0,int(diameter)):
            pi =math.pi
            e = math.e
            kernel1[x,y] = (1/(2*pi*(sigma**2)))*(e**(-1*(((x-r)**2 + (y-r)**2)/(2*(sigma**2)))))        
            
            kernel2[x,y] = (1/(2*pi*k*(sigma**2)))*(e**(-1*((((x-r)**2) + ((y-r)**2))/(2*k*(sigma**2)))))
    print(kernel1)
    print('\n')
    print(kernel2)


   # used for testing purposes for the convolution function
    output1 = signal.convolve2d(img,kernel1, boundary = 'symm', mode ='same')
    output2 = signal.convolve2d(img,kernel2, boundary = 'symm', mode ='same')

    #convolution
    
    #dimensions of the image and the kernels, both kernels are of the same size
    img_width, img_height = img.shape[0], img.shape[1]
    kernel_width, kernel_height = kernel1.shape[0], kernel1.shape[1]
    
    # OPTION 2 convolution
    #output1 = img
    #output2 = img
    #output1 = np.zeros_like(img, dtype = float) # convolution output into zero array for kernel 1
    #output2 = np.zeros_like(img, dtype = float) # convolution output into zero array for kernel 2
    #for rows in range(img.shape[0]):
       #for columns in range(img.shape[1]):
           #if rows<r or columns<r or (rows>=(img.shape[0]-r)) or (columns>=(img.shape[1]-r)):
              #output1[rows,columns] = 0
              #output2[rows,columns] = 0
              #pass
           #for i in range(-1*r,r+1):
               #for j in range(-1*r,r+1):
                    #output1[rows,columns] += (img[x+i,y+j]*kernel1[r+i,r+j])
                    #output2[rows,columns] += (img[x+i,y+j]*kernel2[r+i,r+j])

    # OPTION 1
    #output1 = np.zeros_like(img, dtype = float) # convolution output into zero array for kernel 1
    #output2 = np.zeros_like(img, dtype = float) # convolution output into zero array for kernel 2
    #output1=0
    #output2 =0
    #output1 =img
    #output2 =img
    # Add zero padding to the input image
    #image_padded = np.zeros((img.shape[0] + (r-1), img.shape[1] + (r-1)))   
    #image_padded[-1:1, -1:1] = img
    #for i in range(img.shape[1]):
       #for j in range(img.shape[0]):
           #output1[j,i] = (kernal1[j:j+r , i:i+r]).sum()
           #output2[j,i] =(kernal2[j:j+r , i:i+r]).sum()


    #Applying the XDoG
    XDoG_output = np.zeros_like(output1, dtype = float) 
    for a in range(0,output1.shape[0]):
        for b in range(0,output1.shape[1]):
            XDoG_output[a,b] = (output1[a,b] + p*(output1[a,b] - output2[a,b]))
 
    return XDoG_output


new_img = eXtended_DoG(img, r, sigma)
cv2.imwrite("XDoG_orca.png", new_img)
