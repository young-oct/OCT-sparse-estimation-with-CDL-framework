# -*- coding: utf-8 -*-
# @Time    : 2021-04-27 4:48 p.m.
# @Author  : young wang
# @FileName: quality_compare.py
# @Software: PyCharm

"""this script generates images for the figure 5 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
optimal values of the weighting parameter and lambda"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from skimage.morphology import disk
from skimage.morphology import dilation, erosion
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from skimage import filters
from scipy import ndimage
from tabulate import tabulate


from skimage import feature
from pytictoc import TicToc
from scipy import signal


# Module level constants
eps = 1e-14

def getWeight(s, lmbda, speckle_weight, Paddging = True, opt_par={}):

    l2f, snorm = processing.to_l2_normed(s)

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    rvmin,vmax = 65,115
    x = processing.from_l2_normed(xnorm, l2f)
    x_log = 10 * np.log10(abs(x)**2)
    x_log = processing.imag2uint(x_log, rvmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= rvmin, 0, x_log)

    W = dilation(x_log, disk(5))
    W = erosion(W, disk(5))

    W = np.where(W > 0, speckle_weight, 1)

    # remove residual noise with the median filter,
    # with a kernel size of 5
    W = ndimage.median_filter(W, size=5)

    if Paddging == True:
        pad = 10 #
        # find the bottom edge of the mask with canny edge filter
        temp = filters.sobel(W)

        # temp = quality.gaussian_blur(temp)
        # define a pad value
        pad_value = np.linspace(speckle_weight,1,pad)

        for i in range(temp.shape[1]):
            peak,_ = find_peaks(temp[:,i], height=0)
            if len(peak) != 0:
                loc = peak[-1]
                if temp.shape[0] - loc >= pad:
                    W[loc:int(loc+pad),i] = pad_value
            else:
                W[:, i] = W[:,i]
    else:
        W = W

    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W

def make_sparse_representation(s, lmbda, speckle_weight):
    ''' s -- 2D array of complex A-lines with dims (width, depth)
    '''
    # l2 norm data and save the scaling factor

    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weight factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    W = np.roll(getWeight(s, 0.05,speckle_weight,Paddging = True, opt_par = opt_par), np.argmax(D), axis=0)
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    xnorm = b.solve().squeeze() + eps
    # calculate sparsity
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    ## Convert back from normalized
    x = processing.from_l2_normed(xnorm, l2f)
    return(x)

if __name__ == '__main__':


    #Image processing and display paramaters
    speckle_weight = 0.1
    lmbda = 0.05
    rvmin = 65  # dB
    vmax = 115  # dB

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    file_name = 'finger'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)

    x = make_sparse_representation(s, lmbda, speckle_weight)

    # Generate log intensity arrays
    s_log = 10 * np.log10(abs(s)**2)
    x_log = 10 * np.log10(abs(x)**2)

    # Define ROIs
    roi = {}
    width, height = (17, 10)
    roi['artifact'] = [[210, 135, width*2, height*2]]
    roi['background'] = [[270, 20, width*8, height*8]]
    roi['homogeneous'] = [[210, 165, int(width*1.5), int(height*1.5)],
                   [390, 225, width, height]]

    s_intensity = abs(s)**2
    x_intensity = abs(x)**2

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)

    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0], vmax=vmax, vmin=rvmin)

    text = r'${R_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0],  roi['homogeneous'][0][1]+height), xycoords='data',
                xytext=(roi['homogeneous'][0][0]-50, roi['homogeneous'][0][1]+50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${R_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0],  roi['homogeneous'][1][1]+height), xycoords='data',
                xytext=(roi['homogeneous'][1][0]-50, roi['homogeneous'][1][1]+50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    ax.set_axis_off()
    ax.set_title('reference')

    ax.set_aspect(s_log.shape[1] / s_log.shape[0])

    for i in range(len(roi['artifact'])):
        for j in annotation.get_artifact(*roi['artifact'][i]):
            ax.add_patch(j)
    for i in range(len(roi['background'])):
        for j in annotation.get_background(*roi['background'][i]):
            ax.add_patch(j)
    for i in range(len(roi['homogeneous'])):
        for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
            ax.add_patch(j)

    textstr = '\n'.join((
        r'${SNR_{{R_1}/B}}$''\n'
        r'%.1f dB' % (quality.SNR(ho_s_1,ba_s)),
        r'${C_{{R_1}/B}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_s_1, ar_s)),
        r'${C_{{R_1}/{R_2}}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_s_1, ho_s_2)),
        r'${CNR_{{R_1}/A}}$''\n'
        r'%.1f dB' % (quality.CNR(ho_s_1,ar_s)),
        r'${gCNR_{{R_1}/A}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_s_1, ar_s,N = max(np.size(ho_s_1),np.size(ar_s))))
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=23,
            verticalalignment='top', fontname='Arial', color='red')

    ax = fig.add_subplot(gs[1])

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0], vmax=vmax, vmin=rvmin)

    text = r'${R_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0],  roi['homogeneous'][0][1]+height), xycoords='data',
                xytext=(roi['homogeneous'][0][0]-50, roi['homogeneous'][0][1]+50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${R_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0],  roi['homogeneous'][1][1]+height), xycoords='data',
                xytext=(roi['homogeneous'][1][0]-50, roi['homogeneous'][1][1]+50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    ax.set_title('𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda, speckle_weight))

    ax.set_axis_off()
    for i in range(len(roi['artifact'])):
        for j in annotation.get_artifact(*roi['artifact'][i]):
            ax.add_patch(j)
    for i in range(len(roi['background'])):
        for j in annotation.get_background(*roi['background'][i]):
            ax.add_patch(j)
    for i in range(len(roi['homogeneous'])):
        for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
            ax.add_patch(j)

    textstr = '\n'.join((
        r'${SNR_{{R_1}/B}}$''\n'
        r'%.1f dB' % (quality.SNR(ho_x_1,ba_x)),
        r'${C_{{R_1}/B}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_x_1, ar_x)),
        r'${C_{{R_1}/{R_2}}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_x_1, ho_x_2)),
        r'${CNR_{{R_1}/A}}$''\n'
        r'%.1f dB' % (quality.CNR(ho_x_1,ar_x)),
        r'${gCNR_{{R_1}/A}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_x_1, ar_x,N = max(np.size(ho_x_1),np.size(ar_x))))
    ))
    ax.text(0.05,  0.95, textstr, transform=ax.transAxes, fontsize=23,
            verticalalignment='top', fontname='Arial', color='red')
    plt.tight_layout()
    plt.show()


    # plt.figure()
    # M = getWeight(0.05,speckle_weight=0.1,Paddging=True)
    # im = plt.imshow(M.squeeze())
    # plt.colorbar(im)
    # plt.show()

    # table formant original then sparse
    table = [['SNR between homogeneous and background ROIs', quality.SNR(ho_s_1,ba_s), quality.SNR(ho_x_1,ba_x)],
             ['Contrast between homogeneous and artifact ROIs', quality.Contrast(ho_s_1,ar_s),quality.Contrast(ho_x_1,ar_x)],
             ['Contrast between homogeneous and background ROIs', quality.Contrast(ho_s_1,ba_s), quality.Contrast(ho_x_1,ba_x)],
             ['CNR between homogeneous and artifact ROIs',  quality.CNR(ho_s_1, ar_s), quality.CNR(ho_x_1, ar_x)],
             ['CNR between homogeneous and background ROIs', quality.CNR(ho_s_1, ba_s), quality.CNR(ho_x_1, ba_x)],
             ['Contrast between homogeneous 1 and homogeneous 2 ROIs', quality.Contrast(ho_s_1, ho_s_2), quality.Contrast(ho_x_1, ho_x_2)]]

    print(tabulate(table, headers=['IQA', 'Original image', 'Deconvolved image'],
                   tablefmt='fancy_grid', floatfmt='.2f', numalign='right'))

