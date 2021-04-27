# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 7:52 p.m.
# @Author  : young wang
# @FileName: quality.py
# @Software: PyCharm

import numpy as np

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

def SNR(roi_h):

    '''compute the SNR of a given homogenous region

    SNR = 10*log10(ur/σr^2)

    μr and σ2^2 represent the mean and the variance of
    selected homogeneous ROI [pixel value]

    Reference:
    Stochastic speckle noise compensation in optical
    coherence tomography using non-stationary spline-based
    speckle noise modelling  doi: 10.1364/BOE.4.001769

    parameters
    ----------
    roi_h: array_like
    homogeneous region

    '''
    mean_h = np.mean(roi_h)
    var_h = np.var(roi_h)

    with np.errstate(divide='ignore'):
        snr = 10*np.log10(mean_h / var_h)
    return snr


def CNR(roi_h,roi_b):

    '''compute the CNR between homogeneous and region free of
    structure

    CNR = 10*log((|uh-ub|/sqrt(0.5*(σh^2+σb^2)))

    uh and σh^2 represent the mean and the variance of
    selected homogeneous ROI [pixel value]

    ub and σb^2 represent the mean and the variance of
    selected background ROI [pixel value]

    Reference:
    OCT Image Restoration Using Non-Local Deep Image Prior
    https://doi.org/10.3390/electronics9050784

    parameters
    ----------
    roi_h: array_like
    homogeneous region
    roi_b: array_like
    region free of structure
    '''

    h_mean = np.mean(roi_h)
    b_mean = np.mean(roi_b)

    h_var = np.var(roi_h)
    b_var = np.var(roi_b)
    with np.errstate(divide='ignore'):

        cnr = np.abs(h_mean - b_mean) / np.sqrt(0.5 * (h_var + b_var))

    return 10*np.log10(cnr)

def MIR(roi_1,roi_2):
    '''Mean intensity ratio (MIR) measures the ratio in
    image intensity between a bright and dim structural
    region within an image

    MIR = u1/u2

    u1 and u2 represent the mean
    selected homogeneous ROIs [pixel value]

    '''

    with np.errstate(divide='ignore'):

        mir = np.mean(roi_1)/np.mean(roi_2)

    return mir