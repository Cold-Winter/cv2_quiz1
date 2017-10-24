from scipy.misc.pilutil import imshow
from PIL import Image
from scipy.ndimage import imread
import numpy as np
import math
import time

image = imread('../Images/15088.jpg')
histogram = np.zeros((256),dtype='float32')

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        histogram[image[i][j]] += 1
histogram = histogram / histogram.sum()

histogram = histogram+1e-8    


max = 0
T = 0

for i in range(len(histogram)):
    entropy_a = 0
    for j in range(i):
        prob = histogram[j] / histogram[0:i].sum()
        entropy_a -= prob * (math.log(prob))
    entropy_b = 0
    for j in range(i,histogram.shape[0]):
        prob = histogram[j] / histogram[i:histogram.shape[0]].sum()
        entropy_b -= prob * (math.log(prob))
    entropy = entropy_a+entropy_b
    if entropy >= max:
        max = entropy
        T = i
print max
print T
binary = np.zeros((image.shape[0],image.shape[1]))
binary[image>T]=255
imshow(np.concatenate((image,binary),axis=1))
    
    
        