# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:54 p.m.
# @Author  : young wang
# @FileName: processing.py
# @Software: PyCharm
import numpy as np
from sporco import prox
import pickle
from pathlib import Path
from skimage.morphology import disk,square,star,diamond,octagon
from skimage.morphology import dilation, erosion
from skimage import filters
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from numpy.fft import fft, fftshift, ifft
from scipy import signal
import numpy as np
import pickle
from sporco.admm import cbpdn




# Module level constants
eps = 1e-14
dwell = 20
lmbda = 1e-1

def Aline_R(data,start):
    A_line = ifft(data, axis=1)
    return A_line[dwell * start:dwell * (start + 512), -350:-20].T

def Aline_G(data,start,std):
    window = signal.windows.gaussian(data.shape[1], std=std)
    temp = data*window
    A_line = ifft(temp, axis=1)
    return A_line[dwell * start:dwell * (start + 512), -350:-20].T

def Aline_H(data,start):
    window = np.hanning(data.shape[1])
    temp = data*window
    A_line = ifft(temp, axis=1)
    return A_line[dwell * start:dwell * (start + 512), -350:-20].T

def mean_remove(s,decimation_factor):
    s = s - np.mean(s, axis=1)[:, np.newaxis]
    # (3) sample every decimation_factor line,
    s = s[:, ::decimation_factor]
    return s

def load_raw(file_path):
    if Path(file_path).is_file():
        # with open(file_path, 'rb') as f:
            # raw = pickle.load(f)
            # f.close()
        temp = np.load(file_path)
        raw = temp['arr_1']
        return raw

    else:
        raise Exception("Dataset %s not found" % file_path)

def imag2uint(data, vmin, vmax):
    data = np.clip(data, vmin, vmax)
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals

def to_l2_normed(s):
    l2f = prox.norm_l2(s, axis=0).squeeze()
    return (l2f, s / l2f)

def from_l2_normed(s, l2f):
    return (s * l2f)

def load_data(dataset_name, decimation_factor, data_only=False):
    # check if such file exists
    S_PATH = '../data/' + dataset_name

    if data_only == False:
        D_PATH = '../data/PSF/' + dataset_name
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
    else:
        if Path(S_PATH).is_file():
            # load data & dictionary
            with open(S_PATH, 'rb') as f:
                s = pickle.load(f).T
                f.close()
            # (2) remove background noise: minus the frame mean
            s = s - np.mean(s, axis=1)[:, np.newaxis]
            # (3) sample every decimation_factor line,
            s = s[:, ::decimation_factor]
            return s
        else:
            raise Exception("Dataset %s not found" % dataset_name)


def getWeight(s, D, w_lmbda, speckle_weight, Paddging=True, opt_par={},Ear = False):
    l2f, snorm = to_l2_normed(s)

    b = cbpdn.ConvBPDN(D, snorm, w_lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    rvmin, vmax = 5, 55
    x = from_l2_normed(xnorm, l2f)
    x_log = 10 * np.log10(abs(x) ** 2)
    x_log = imag2uint(x_log, rvmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= rvmin, 0, x_log)
    W = dilation(x_log, square(3))
    W = erosion(W, square(3))
    W = np.where(W > 0, speckle_weight, 1)

    if Ear == True:

        W = filters.median(W, square(7))

    else:

        W = filters.median(W, square(17))

    if Paddging == True:
        pad = 20  #
        # find the bottom edge of the mask with Sobel edge filter

        temp = filters.sobel(W)
        # temp = gaussian_filter(temp, 3)

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
        pass

    W = filters.median(W, square(7))
    W = gaussian_filter(W, sigma=0.5)
    W = filters.median(W, square(12))

    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W

def make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight, Line=False, index=None, Mask=False, Ear =False):
    ''' s -- 2D array of complex A-lines with dims (width, depth)
    '''
    # l2 norm data and save the scaling factor
    l2f, snorm = to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weight factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    # if Ear == True:
    #     w_lambda = 0.0
    # else:
    #     pass

    W = np.roll(getWeight(s, D, w_lmbda, speckle_weight, Paddging=True, opt_par=opt_par,Ear = Ear), np.argmax(D), axis=0)
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    xnorm = b.solve().squeeze() + eps
    # calculate sparsity
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    if Line == False and Mask == False:
        ## Convert back from normalized
        x = from_l2_normed(xnorm, l2f)
        return (x)
    elif Line == True and Mask == False:
        assert index != None and 0 <= index <= s.shape[1]
        x = from_l2_normed(xnorm, l2f)
        x_line = abs(xnorm[:, index])
        return x, x_line
    elif Line == False and Mask == True:
        x = from_l2_normed(xnorm, l2f)
        W_mask = np.roll(W, -np.argmax(D), axis=0).squeeze()
        return x, W_mask
    else:
        x = from_l2_normed(xnorm, l2f)
        x_line = abs(xnorm[:, index])
        W_mask = np.roll(W, -np.argmax(D), axis=0).squeeze()
        return x, x_line, W_mask
