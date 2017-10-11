import numpy as np
from scipy.ndimage import imread
from scipy.misc.pilutil import imshow
from scipy.signal import convolve2d as conv
from scipy import *


def gaussFilter(sigma,window = 3):
    '''
        This method is used to create a gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    kernel = zeros((window,window))
    c0 = window // 2

    for x in range(window):
        for y in range(window):
            r = hypot((x-c0),(y-c0))
            val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
            kernel[x,y] = val
    return kernel / kernel.sum()


def duplicate_37(img,padding = 3):
    padimg_x = np.concatenate((np.zeros((padding,img.shape[1])),img,np.zeros((padding,img.shape[1]))),axis=0)
    padimgaround = np.concatenate((np.zeros((padimg_x.shape[0],padding)),padimg_x,np.zeros((padimg_x.shape[0],padding))),axis=1)
    dup_37 = np.zeros((img.shape[0],img.shape[1],37))
    index = 0
    for i in range(circlemask.shape[0]):
        for j in range(circlemask.shape[1]):
            if circlemask[i][j]>0:
                dup_37[:,:,index] = padimgaround[i:i+img.shape[0], j:j+img.shape[1]]
                index += 1
    return dup_37

# non maximum suppression with distance of gravity center
def susan(img,dupimg_37, dx,dy,thresh = 1.1, t = 32,thelta=0.5):
    distance = np.exp(-np.power((dupimg_37 - img[:,:,None]) / t, 6))
    n_r0 = distance.sum(axis = 2)

    g = thelta * n_r0.max()
    R_r0 = (n_r0 <= g).astype(int) * (g - n_r0)

    x_sum = np.reshape(np.dot(distance, dx), (img.shape[0],img.shape[1]))
    y_sum = np.reshape(np.dot(distance, dy), (img.shape[0],img.shape[1]))
    
    dist_x = x_sum / (n_r0 + 1e-8)
    dist_y = y_sum / (n_r0 + 1e-8)

    dist2 = np.sqrt(dist_x * dist_x + dist_y * dist_y)
    mask = np.zeros((dist2.shape[0],dist2.shape[1]))
#     low = dist2.min()
#     high = dist2.max()
#     img = (dist2-low) * 1.0 / (high - low)
    mask[dist2>=thresh]=255
    return mask


circlemask = np.asarray([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]],
                      dtype = np.int)
circleflat = circlemask.flatten()
 
fx = np.reshape(np.repeat(np.asarray(range(-3, 4), dtype = np.float32), 7), (7,7))
fy = np.transpose(fx)

dx = np.asarray([fx.flatten().tolist()[i] for i in range(circleflat.shape[0]) if circleflat[i] > 0])
dy = np.asarray([fy.flatten().tolist()[i] for i in range(circleflat.shape[0]) if circleflat[i] > 0])
dx = np.reshape(dx, (dx.shape[0], 1))
dy = np.reshape(dy, (dy.shape[0], 1))
       
image = imread('../Images/susan_input1.png')
dup_37 = duplicate_37(image)
#4.1
mask1 = susan(image,dup_37,dx,dy,1.2)
imshow(mask1)
image = imread('../Images/susan_input2.png')
dup_37 = duplicate_37(image)
#4.2
mask2 = susan(image,dup_37,dx,dy,1.2)
imshow(mask2)
'''
We can draw a conclusion the SUSAN operator may not be robust for the noise of this image
'''

#4.3
image = imread('../Images/susan_input1.png')
gaussion = gaussFilter(sigma=1,window=10)

noiseimg = conv(image,gaussion,mode='same')
imshow(noiseimg)
mask3 = susan(image,dup_37,dx,dy,1.2)
imshow(mask3)
'''
SUSAN operator may be robust for smooth images.
'''



























