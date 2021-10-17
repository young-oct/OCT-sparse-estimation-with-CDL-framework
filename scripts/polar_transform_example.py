# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:49 p.m.
# @Author  : young wang
# @FileName: image_compare.py
# @Software: PyCharm

'''From left to right: OCT images of a middle ear,
 index finger (palmar view), index finger (side view),
  and onion slice. The white arrow indicates the sidelobe
  artifacts caused by the PSF convolution'''


import polarTransform
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    plt.close('all')
        
    imagec=plt.imread('misc/llama.jpg') 
    image=np.mean(imagec,axis=2)
    image[10,10:415]=255
    image[415,10:415]=255
    image[10:415,10]=255
    image[10:415,415]=255
    plt.figure()
    plt.imshow(image,cmap='gray')
    
    opening_angle=60 #deg
    polarImage, ptSettings = polarTransform.convertToCartesianImage(image.T, 
                                                                    initialRadius=200, 
                                                                    finalRadius=1024, 
                                                                    initialAngle=-opening_angle*np.pi/360, 
                                                                    finalAngle=opening_angle*np.pi/360)
    plt.figure()
    plt.imshow(polarImage.T,cmap='gray') 
               
    