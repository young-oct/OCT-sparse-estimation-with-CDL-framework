# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:49 p.m.
# @Author  : young wang
# @FileName: image_compare.py
# @Software: PyCharm

'''From left to right: OCT images of a middle ear,
 index finger (palmar view), index finger (side view),
  and onion slice. The white arrow indicates the sidelobe
  artifacts caused by the PSF convolution'''


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from misc import processing
from scipy.ndimage import median_filter
import polarTransform

# Module level constants
eps = 1e-14

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    file_name = ['ear', 'finger', 'nail', 'onion']
    title_name = [r'(a) middle ear', r'(b) index finger (palmar view)', r'(c) index finger (side view)', r'(d) onion slice']

    original = []
    sparse = []

    lmbda = [0.05,0.03,0.02,0.04]
    w_lmbda = 0.05
    speckle_weight = 0.1
    rvmin, vmax = 5, 55 #dB

    for i in range(len(file_name)):
        Ear = False
        # Load the example dataset
        s, D = processing.load_data(file_name[i], decimation_factor=20)
        # l2 norm data and save the scaling factor
        l2f, snorm = processing.to_l2_normed(s)

        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

        # obtain weighting mask
        if file_name[i] == 'ear':
            Ear = True
        else:
            pass
        x = processing.make_sparse_representation(s, D, lmbda[i],w_lmbda, speckle_weight,Ear= Ear)

        x_log = 20 * np.log10(abs(x))
        s_log = 20 * np.log10(abs(s))
        original.append(s_log)
        sparse.append(x_log)

    x_head = [300, 200, 240, 250]
    y_head = [110, 125, 170, 120]

    x_end = [350, 150, 190, 190]
    y_end = [90, 105, 150, 100]

    aspect = original[0].shape[1]/original[0].shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=True, figsize=(16, 9),constrained_layout=True )

    cartesianImage=x_log
    
    for i in range(len(file_name)):
        title = '\n'.join((title_name[i],r'$𝜆$ = %.2f,$W$ = %.1f' % (lmbda[i], speckle_weight)))

        ax[0, i].set_title(title,fontsize=20)
        ax[0, i].imshow(original[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin,interpolation='none')
        
        
        #ax[0, i].annotate('', xy=(x_head[i], y_head[i]), xycoords='data',
        #                  xytext=(x_end[i], y_end[i]), textcoords='data',
        #                  arrowprops=dict(facecolor='white', shrink=0.05),
        #                  horizontalalignment='right', verticalalignment='top',
        #                  )

        ax[1, i].imshow(sparse[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin,interpolation='none')
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    plt.show()

    fig.savefig('../Images/image_compare.svg',
                dpi = 1200,
                transparent=True,format = 'svg')

    
    from numpy import pi
    #plt.close('all')
    ear_image=sparse[0]
    ear_image[0,:]=vmax
    ear_image[-1,:]=vmax
    ear_image[:,0]=vmax
    ear_image[:,-1]=vmax
    ear_image = median_filter(ear_image, size=(2, 2))
    for i in range(ear_image.shape[0]):
        for j in range(ear_image.shape[1]):
            if ear_image[i,j]<rvmin:
                ear_image[i,j]=rvmin
            if ear_image[i,j]>vmax:
                ear_image[i,j]=vmax
                
        
    
    opening_angle=60 #deg
    polarImage, ptSettings = polarTransform.convertToCartesianImage(ear_image.T, initialRadius=300, finalRadius=812, initialAngle=-opening_angle*pi/360, finalAngle=opening_angle*pi/360)
    plt.figure()
    plt.imshow(polarImage.T[::-1,:], 'gray',aspect=aspect,vmax=vmax, interpolation='none', vmin=rvmin, origin='lower')
    plt.figure()
    plt.imshow(ear_image, 'gray',aspect=aspect,vmax=vmax, vmin=rvmin, interpolation='none', origin='lower')