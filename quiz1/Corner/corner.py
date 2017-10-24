from scipy.ndimage import imread
from scipy.misc.pilutil import imshow
import numpy as np
from scipy import *
from scipy.signal import convolve2d as conv
import time

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

def gaussFilterOneDim(sigma,window = 3):
    '''
        This method is used to create a one dimension gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    kernel = zeros((window,))
    c0 = window // 2

    for x in range(window):
        r = (x-c0)
        val = (1.0/(sqrt(2*pi)*sigma))*exp(-(r*r)/(2*sigma*sigma))
        kernel[x] = val
    return kernel 

def conv1d(img,gdis,padding = 1,window = 3):
    padimg = np.concatenate((np.zeros((padding,img.shape[1])),img,np.zeros((padding,img.shape[1]))),axis=0)
    img3d = np.zeros((img.shape[0],img.shape[1],window),dtype='float32')
    for i in range(window):
        img3d[:,:,i]=padimg[i:padimg.shape[0]-window+i+1,]
    filteimg = np.dot(img3d,gdis)
    return filteimg
def gradientImg(img):
    padimg = np.concatenate((np.zeros((1,img.shape[1])),img),axis=0)
    gradientImg = padimg[1:padimg.shape[0],] - padimg[0:padimg.shape[0]-1,]
    return gradientImg
    

def basecorner(img,thresh=0.56):
    gradkernel = np.asarray([-1,0,1],dtype='float32')
    dx = conv1d(img,gradkernel,1,3)
    dxx = conv1d(dx,gradkernel,1,3)
    dy = np.transpose(conv1d(np.transpose(img),gradkernel,1,3))
    dyy = np.transpose(conv1d(np.transpose(dy),gradkernel,1,3))
    dxy = np.transpose(conv1d(np.transpose(dx),gradkernel,1,3))
    hessian = np.zeros((img.shape[0],img.shape[1],2,2),dtype='float32')
    hessian[:,:,0,0] = dxx
    hessian[:,:,0,1] = dxy
    hessian[:,:,1,0] = dxy
    hessian[:,:,1,1] = dyy
    eaignvalue = np.linalg.eigvals(hessian)

    mask = np.zeros((img.shape[0],img.shape[1]))+255

    min = eaignvalue.min()
    max = eaignvalue.max()
    eaignvalue = (eaignvalue-min)/(max-min)
    mask[eaignvalue[:,:,0]<=thresh]=0
    mask[eaignvalue[:,:,1]<=thresh]=0
    return mask

def harriscorner(img,thresh=0.2,sigma=1,window=3):
    gaussion = gaussFilter(sigma,window)
    gradkernel = np.asarray([-1,0,1],dtype='float32')
    dx = conv1d(img,gradkernel,1,3)
    dy = np.transpose(conv1d(np.transpose(img),gradkernel,1,3))
    dx2 = dx*dx
    dy2 = dy*dy
    dxy = dx*dy
    
    dx2 = conv(dx2,gaussion,mode='same')
    dy2 = conv(dy2,gaussion,mode='same')
    dxy = conv(dxy,gaussion,mode='same')
    
    hessian = np.zeros((img.shape[0],img.shape[1],2,2),dtype='float32')
    hessian[:,:,0,0] = dx2
    hessian[:,:,0,1] = dxy
    hessian[:,:,1,0] = dxy
    hessian[:,:,1,1] = dy2
    cornerness = np.linalg.det(hessian) - 0.04 * np.trace(hessian, axis1 = hessian.ndim-2, axis2 = hessian.ndim-1)
#     eaignvalue = np.linalg.eigvals(hessian)
# 
    mask = np.zeros((img.shape[0],img.shape[1]))+255
    min = cornerness.min()
    max = cornerness.max()
    cornerness = (cornerness-min)/(max-min)
    mask[cornerness<=thresh]=0
    return mask
def harriscornerslow(img,thresh=0.2,sigma=1,window=3):
    gaussion = gaussFilter(sigma,window)
    gradkernel = np.asarray([-1,0,1],dtype='float32')
    dx = conv1d(img,gradkernel,1,3)
    dy = np.transpose(conv1d(np.transpose(img),gradkernel,1,3))
    dx2 = dx*dx
    dy2 = dy*dy
    dxy = dx*dy
    
    dx2 = conv(dx2,gaussion,mode='same')
    dy2 = conv(dy2,gaussion,mode='same')
    dxy = conv(dxy,gaussion,mode='same')
    
    hessian = np.zeros((img.shape[0],img.shape[1],2,2),dtype='float32')
    hessian[:,:,0,0] = dx2
    hessian[:,:,0,1] = dxy
    hessian[:,:,1,0] = dxy
    hessian[:,:,1,1] = dy2
    #cornerness = np.linalg.det(hessian) - 0.04 * np.trace(hessian, axis1 = hessian.ndim-2, axis2 = hessian.ndim-1)
#     eaignvalue = np.linalg.eigvals(hessian)
    eaignvalue = np.linalg.eigvals(hessian)
    cornerness = np.prod(eaignvalue, axis = 2) - 0.04 * np.sum(eaignvalue, axis = 2)
    mask = np.zeros((img.shape[0],img.shape[1]))+255
    min = cornerness.min()
    max = cornerness.max()
    cornerness = (cornerness-min)/(max-min)
    mask[cornerness<=thresh]=0

    return mask
image = imread('../Images/input3.png')
if len(image.shape)==3:
    image = image[:,:,0]
start = time.time()
maskbase = basecorner(image)
print time.time()-start
start = time.time()
maskharrisslow = harriscornerslow(image)
print time.time()-start
start = time.time()
maskharris = harriscorner(image)
print time.time()-start
imshow(np.concatenate((maskbase,maskharris,maskharrisslow,image),axis=1))
            
        
        
        


    
