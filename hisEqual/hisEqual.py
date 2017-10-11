from scipy.misc.pilutil import imshow
from PIL import Image
from scipy.ndimage import imread
import numpy as np
import math
import time


def hisEqual(imname):
    img = imread(imname)
    histogram = np.zeros((256),dtype='float32')
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1
    histogram = histogram / histogram.sum()
    histcumsum = np.cumsum(histogram)
    map = np.round(255.0*histcumsum)
    hisEqualImg = map[img]
    return NormImage(hisEqualImg)
def clipping(imname,a=50,b=150,beta=2):
    img = imread(imname)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>=a and img[i][j]<b:
                img[i][j] = beta*(img[i][j]-a)
            elif img[i][j]>=b:
                img[i][j] = beta*(b-a)
            else:
                img[i][j] = 0
    return NormImage(img)  
def NormImage(img):
    low = img.min()
    high = img.max()
    img = (img-low) * 1.0 / (high - low)
    return img * 255
def rangCompress(imname,c=1):
    img = imread(imname)
    map = c * np.log10(1 + np.asarray(range(256), dtype = np.float32))
    img = map[img]
    #return NormImage(img)
    return img
    

    
#     equImg = img[]
imname = '../Images/15088.jpg'
image = imread(imname)       
#5.1
hisEqualImg = hisEqual(imname)
#5.2

'''
The results are much more clear than the original image, since it clipping the low value of pixel to 0 and make large value of pixel larger 
'''
clip = clipping(imname)
imshow(np.concatenate((image,hisEqualImg,clip),axis=1))
#5.3
'''
If c is larger, the image will be brighter  
''' 
clist = [1,10,100,100]
imageout = []
for c in clist:
    imageout.append(rangCompress(imname,c))
imshow(np.concatenate(imageout,axis=1))
#

