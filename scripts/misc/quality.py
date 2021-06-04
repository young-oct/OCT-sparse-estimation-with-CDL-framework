# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 7:52 p.m.
# @Author  : young wang
# @FileName: quality.py
# @Software: PyCharm

import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from misc.processing import imag2uint


def gaussian_blur(noisy, sigma=0.5):
    out = gaussian(noisy, sigma=sigma, output=None, mode='nearest', cval=0,
                   multichannel=None, preserve_range=False, truncate=4.0)
    return (out)

def ROI(x, y, width, height,s):
    '''obtain the ROI from the standard layout [330x512]

    parameters
    ----------
    s has the standard dimension [330x512]
    y is defined as the axial direction: 330
    x is defined as the lateral direction: 512
    height refers the increment in the axial direction > 0
    width refers the increment in the lateral direction > 0
    '''
    # fetch ROI
    if height > 0 and width > 0:
        if (x >= 0) and (y >= 0) and (x + width <= s.shape[1]) and (y + height <= s.shape[0]):
            roi = s[y:y + height, x:x + width]
    return roi

def SF(s):
    '''obtain the sparsity fraction from given region of interest

    parameters
    ----------
    i_{mn} represents the matrix of pixel intensities at
    each location \left(m,n\right) in an N by M image patch,
    where im,n0 is the l_0 norm of i_{mn}, i.e.,
    the number of nonzero elements
    '''

    return (1 - np.count_nonzero(s) / s.size)

def SNR(roi_h,roi_b):

    '''compute the SNR of a given homogenous region

    SNR = 10*log10(uh/σb)

    Improving ultrasound images with
    elevational angular compounding based on
    acoustic refraction
    https://doi.org/10.1038/s41598-020-75092-8

    parameters
    ----------
    roi_h: array_like
    homogeneous region

    roi_b: array_like
    background region

    '''

    mean_h = np.mean(roi_h)
    std_b = np.std(roi_b)

    with np.errstate(divide='ignore'):

        snr = 10*np.log10(mean_h/ std_b)

    return snr

def CNR(roi_h,roi_a):

    '''compute the CNR between homogeneous and region free of
    structure

    CNR = 10*log((|uh-ub|/σb)

    Reference:
    Improving ultrasound images with
    elevational angular compounding based on
    acoustic refraction
    https://doi.org/10.1038/s41598-020-75092-8


    parameters
    ----------
    roi_h: array_like
    homogeneous region
    roi_a: array_like
    region free of structure
    '''

    h_mean = np.mean(roi_h)
    a_mean = np.mean(roi_a)

    a_std = np.std(roi_a)

    with np.errstate(divide='ignore'):

        cnr = abs(h_mean - a_mean) / a_std

    return 10*np.log10(cnr)

def Contrast(region_h, region_b):

    h_mean = np.mean(region_h)
    b_mean = np.mean(region_b)

    with np.errstate(divide='ignore'):
        contrast = h_mean / b_mean

    return 10*np.log10(contrast)
    # return contrast

def log_gCNR(region_h, region_b, improvement = False):
    assert np.size(region_h) == np.size(region_b), \
        'size of image patch'

    if improvement == True:
        region_h = median_filter(region_h,size=(3,3))
        region_b = median_filter(region_b,size=(3,3))
    else:
        pass

    region_h = np.ravel(region_h)
    region_b = np.ravel(region_b)


    N = 256

    rvmin, vmax = 5, 55 #dB

    # in histogram when density flag is set to be true, the integral is
    # 1 instead of the cumulative PDF, to address this, bin width needs to
    # be the same

    log_h1 = imag2uint(10*np.log10(region_h), rvmin, vmax)
    log_h2 = imag2uint(10*np.log10(region_b), rvmin, vmax)

    min_val, max_val = 0, 255

    h_hist, edge = np.histogram(log_h1, bins=N, range=(min_val, max_val), density=True)
    h_hist = h_hist * np.diff(edge)
    b_hist, edge = np.histogram(log_h2, bins=N, range=(min_val, max_val), density=True)
    b_hist = b_hist * np.diff(edge)

    ovl = 0

    for i in range(0,N):

        ovl += min(h_hist[i], b_hist[i])

    return 1 - ovl

    

