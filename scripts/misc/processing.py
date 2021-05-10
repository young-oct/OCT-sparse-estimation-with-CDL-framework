# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:54 p.m.
# @Author  : young wang
# @FileName: processing.py
# @Software: PyCharm
import numpy as np
from sporco import prox
import pickle
from pathlib import Path
from scipy import ndimage
from skimage.morphology import disk
from sporco.admm import cbpdn
from skimage.morphology import dilation, erosion
from skimage import filters
from scipy.signal import find_peaks

# Module level constants
eps = 1e-14


def imag2uint(data, vmin, vmax):
    data = np.clip(data, vmin, vmax)
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals


def to_l2_normed(s):
    l2f = prox.norm_l2(s, axis=0).squeeze()
    return (l2f, s / l2f)


def from_l2_normed(s, l2f):
    return (s * l2f)


def load_data(dataset_name, decimation_factor):
    # check if such file exists
    S_PATH = '../Data/' + dataset_name
    D_PATH = '../Data/PSF/' + dataset_name
    if Path(S_PATH).is_file() and Path(D_PATH).is_file():
        # load data & dictionary
        with open(S_PATH, 'rb') as f:
            s = pickle.load(f).T
            f.close()

        with open(D_PATH, 'rb') as f:
            D = pickle.load(f)
            f.close()

        # (2) remove background noise: minus the frame mean
        s = s - np.mean(s, axis=1)[:, np.newaxis]
        # (3) sample every decimation_factor line,
        s = s[:, ::decimation_factor]
        return (s, D)
    else:
        raise Exception("Dataset %s not found" % dataset_name)


def getWeight(s, D, lmbda, speckle_weight, Paddging=True, opt_par={}):
    l2f, snorm = to_l2_normed(s)

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    rvmin, vmax = 65, 115
    x = from_l2_normed(xnorm, l2f)
    x_log = 10 * np.log10(abs(x) ** 2)
    x_log = imag2uint(x_log, rvmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= rvmin, 0, x_log)

    W = dilation(x_log, disk(5))
    W = erosion(W, disk(5))

    W = np.where(W > 0, speckle_weight, 1)

    # remove residual noise with the median filter,
    # with a kernel size of 5
    W = ndimage.median_filter(W, size=5)

    if Paddging == True:
        pad = 10  #
        # find the bottom edge of the mask with canny edge filter
        temp = filters.sobel(W)

        # temp = quality.gaussian_blur(temp)
        # define a pad value
        pad_value = np.linspace(speckle_weight, 1, pad)

        for i in range(temp.shape[1]):
            peak, _ = find_peaks(temp[:, i], height=0)
            if len(peak) != 0:
                loc = peak[-1]
                if temp.shape[0] - loc >= pad:
                    W[loc:int(loc + pad), i] = pad_value
            else:
                W[:, i] = W[:, i]
    else:
        W = W

    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W


def make_sparse_representation(s, D, lmbda, speckle_weight, Line=False, index=None):
    ''' s -- 2D array of complex A-lines with dims (width, depth)
    '''
    # l2 norm data and save the scaling factor

    l2f, snorm = to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weight factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    W = np.roll(getWeight(s, D, 0.05, speckle_weight, Paddging=True, opt_par=opt_par), np.argmax(D), axis=0)
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    xnorm = b.solve().squeeze() + eps
    # calculate sparsity
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    if Line == False:
        ## Convert back from normalized
        x = from_l2_normed(xnorm, l2f)
        return (x)
    else:
        assert index != None and 0 <= index <= s.shape[1]
        x = from_l2_normed(xnorm, l2f)
        x_line = abs(xnorm[:, index])
        return x, x_line
