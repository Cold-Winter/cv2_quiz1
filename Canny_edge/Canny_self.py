from PIL import Image
from scipy.ndimage import imread
from scipy.signal import convolve2d as conv
from scipy.misc.pilutil import imshow
from scipy import *
from scipy import ndimage as ndi
import numpy as np


class Canny:
    def __init__(self,imname,sigma,thresLow = 15,thresHigh = 40,window=11):
        self.image = imread(imname)
        if len(self.image.shape)==3:
            self.image = self.image[:,:,0]
        gaussion = self.gaussFilterOneDim(sigma,window)
        gaussDer = self.gaussDerive(sigma,window)
#         print gaussion
#         print gaussDer
        
        self.xconv = self.conv1d(self.image,gaussion,window//2,window)
        self.yconv = np.transpose(self.conv1d(np.transpose(self.image),gaussion,window//2,window))

        
        self.xdconv = self.conv1d(self.xconv,gaussDer,window//2,window)
        self.ydconv = np.transpose(self.conv1d(np.transpose(self.yconv),gaussDer,window//2,window))
        

        
        self.magimg = self.magnitude(self.xdconv, self.ydconv)

        
        self.magimg2 = self.nms4angel(self.xdconv,self.ydconv,self.magimg)
        
        #print magimg2
        
        self.threimg = self.apply_hysteresis_threshold(self.magimg2,thresLow,thresHigh).astype(int)

    def showImg(self):
        upchannel = np.concatenate([self.NormImage(img) for img in [self.image,self.xconv,self.yconv,self.xdconv]],axis=1)
        downchannel = np.concatenate([self.NormImage(img) for img in [self.ydconv,self.magimg,self.magimg2,self.threimg]],axis=1)
        showimg = np.concatenate((upchannel,downchannel),axis=0)
        imshow(showimg)
        
    def NormImage(self,img):

        low = img.min()
        high = img.max()
        img = (img-low) * 1.0 / (high - low)
        return img * 255

    def gaussFilter(self,sigma,window = 5):
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
    def gaussFilterOneDim(self,sigma,window = 5):
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
    def gaussDerive(self,sigma,window=3):
        kernel = zeros((window,))
        c0 = window // 2
        for x in range(window):
            r = x-c0
            val = -(r/(sqrt(2*pi)*sigma*sigma*sigma))*exp(-(r*r)/(2*sigma*sigma))
            kernel[x] = val
        return kernel
    def conv1d(self,img,gdis,padding = 2,window = 5):
        padding = window//2
        padimg = np.concatenate((np.zeros((padding,img.shape[1])),img,np.zeros((padding,img.shape[1]))),axis=0)
        img3d = np.zeros((img.shape[0],img.shape[1],window),dtype='float32')
        for i in range(window):
            img3d[:,:,i]=padimg[i:padimg.shape[0]-window+i+1,]
        filteimg = np.dot(img3d,gdis)
        return filteimg
    def magnitude(self,xdconv,ydconv):
        mag = xdconv*xdconv+ydconv*ydconv
        #mag = np.multiply(xdconv,xdconv)+np.multiply(ydconv,ydconv)
        return np.sqrt(mag)
    def nms(self,xdconv,ydconv,magimginput):
        magimg = magimginput.copy()

        theta = np.arctan2(ydconv,xdconv)
#         print theta
        scalartheta = np.rint(theta / pi * 4 + 4).astype(int)
        scalartheta[scalartheta==8]=0
        mask = np.ones((magimg.shape[0],magimg.shape[1]))
        for i in range(1,scalartheta.shape[0]-1):
            for j in range(1,scalartheta.shape[1]-1):
                if scalartheta[i][j] == 0 or scalartheta[i][j] == 4:
                    if not (magimg[i][j]>=magimg[i+1][j] and magimg[i][j]>=magimg[i-1][j]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0
                elif scalartheta[i][j] == 2 or scalartheta[i][j] == 6:
                    if not (magimg[i][j]>=magimg[i][j+1] and magimg[i][j]>=magimg[i][j-1]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0
                elif scalartheta[i][j] == 1 or scalartheta[i][j] == 5:
                    if not (magimg[i][j]>=magimg[i+1][j-1] and magimg[i][j]>=magimg[i-1][j+1]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0
                elif scalartheta[i][j] == 3 or scalartheta[i][j] == 7:
                    if not (magimg[i][j]>=magimg[i+1][j+1] and magimg[i][j]>=magimg[i-1][j-1]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0

#         print np.sum(mask)
#         min = magimg.min()
#         max = magimg.max()
        return self.NormImage(magimg)
    def nms4angel(self,xdconv,ydconv,magimginput):
        magimg = magimginput.copy()

        theta = np.arctan2(ydconv,xdconv)
#         print theta
        scalartheta = np.rint(theta / pi * 2 + 2).astype(int)
        scalartheta[scalartheta==4]=0
        mask = np.ones((magimg.shape[0],magimg.shape[1]))
        for i in range(1,scalartheta.shape[0]-1):
            for j in range(1,scalartheta.shape[1]-1):
                if scalartheta[i][j] == 0 or scalartheta[i][j] == 2:
                    if not (magimg[i][j]>=magimg[i+1][j] and magimg[i][j]>=magimg[i-1][j]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0
                elif scalartheta[i][j] == 1 or scalartheta[i][j] == 3:
                    if not (magimg[i][j]>=magimg[i][j+1] and magimg[i][j]>=magimg[i][j-1]) :
                        magimg[i][j] = 0
                        mask[i][j] = 0

#         print np.sum(mask)
#         min = magimg.min()
#         max = magimg.max()
        return self.NormImage(magimg)    
    
    def apply_hysteresis_threshold(self,image, low, high):
        low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
        mask_low = image > low
        mask_high = image > high
        # Connected components of mask_low
        labels_low, num_labels = ndi.label(mask_low)
        # Check which connected components contain pixels from mask_high
        sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
        connected_to_high = sums > 0
        thresholded = connected_to_high[labels_low]
        return thresholded
    def returnImg(self):
        return self.threimg
            
                            
threimg = Canny('../Images/181079.jpg',1,window=3).showImg()
sigmas = [0.1, 1 ,10, 100]
outputs = []
for sigma in sigmas:
    threimg = Canny('../Images/181079.jpg', sigma).returnImg()
    outputs.append(threimg)
     
imgshow = np.concatenate(outputs, axis = 1)
imshow(imgshow)

