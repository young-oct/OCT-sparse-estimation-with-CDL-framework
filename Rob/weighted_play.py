# -*- coding: utf-8 -*-
# @Time    : 2021-04-19 9:41 a.m.
# @Author  : young wang
# @FileName: rob.py
# @Software: PyCharm

'''this script shows an indirect evidence that the sparse image is super resolution
version of the original image through 1d convolution'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sporco import prox
from sporco.admm import cbpdn
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import feature
from PIL import Image
from scipy.signal import find_peaks


from skimage import feature
from scipy import ndimage, misc

# Module level constants
eps = 1e-14

def imag2uint(data, vmin, vmax):
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals

def gaussian_blur(noisy, sigma=0.5):
    out = gaussian(noisy, sigma=sigma, output=None, mode='nearest', cval=0,
                   multichannel=None, preserve_range=False, truncate=4.0)
    return (out)

def load_data(dataset_name, decimation_factor):
    # define signal & dictionary path
    if dataset_name == 'finger':
        S_PATH = '../Data/ear'
        D_PATH = '../Data/finger_1.05'

        # load signal & dictionary
        with open(S_PATH, 'rb') as f:
            s = pickle.load(f).T
            f.close()

        with open(D_PATH, 'rb') as f:
            D = pickle.load(f)
            f.close()

        # (2) remove background noise: minus the frame mean
        s = s - np.mean(s, axis=1)[:, np.newaxis]

        # Only  keep every 30th line
        s = s[:, ::decimation_factor]
        return (s, D)
    else:
        raise Exception("Dataset %s not found" % dataset_name)


def plot_images(plot_titles, plot_data,
                vmin, vmax, overlay_plots=None,
                suptitle=None):
    assert len(plot_titles) == len(plot_data), 'Length of plot_titles does not match length of plot_data'

    nplots = len(plot_titles)
    fig, axes = plt.subplots(1, nplots, constrained_layout=True, figsize=(16, 9))
    if overlay_plots is None:
        overlay_plots = [None for i in range(len(plot_titles))]

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im, overlay_plot) in enumerate(zip(axes, plot_titles, plot_data, overlay_plots)):
        ax.set_title(title)
        ax.imshow(im, vmax=vmax, vmin=rvmin, cmap='gray')
        if overlay_plot is not None:
            ax.plot(overlay_plot)
            ax.set_axis_off()
    plt.show()


def plot_alines(plot_titles, plot_data,
                suptitle=None):
    assert len(plot_titles) == len(plot_data), 'Length of plot_titles does not match length of plot_data'

    nplots = len(plot_titles)
    fig, axes = plt.subplots(1, nplots, constrained_layout=True, figsize=(16, 9))

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes, plot_titles, plot_data)):
        ax.set_title(title)
        ax.plot(im, lw=0.5)
        # ax.set_ylim(vmin,vmax)

    plt.show()

def to_l2_normed(s):
    l2f = prox.norm_l2(s, axis=0).squeeze()
    return (l2f, s / l2f)

def from_l2_normed(s, l2f):
    return (s * l2f)
#
# def bmp_save(data,path):
#     pixel_vals = (255) * (data - data.min()) / (data.max() - data.min())
#     # pixel_vals = 255 * (data - rvmin) / (vmax - rvmin)
#
#     pixel_vals = np.where(pixel_vals <= 150,0,pixel_vals)
#
#     # pixel_vals = np.uint8(np.around(pixel_vals))
#     pixel_vals = np.uint8(pixel_vals)
#
#     image = Image.fromarray(pixel_vals)
#     image.save(path)
#     # # pixel_vals = np.uint8(np.around(255 * (data - rvmin) / (vmax - rvmin), 0))
#     # # pixel_vals = pixel_vals.astype(np.uint8)
#     # image = Image.fromarray(pixel_vals)
#     # image.save(path)

def pixel_norm(data):
    pixel_vals = (255) * (data - data.min()) / (data.max() - data.min())
    return pixel_vals

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 12
        }
    )

    index = 200
    # Load the example dataset
    s, D = load_data('finger', decimation_factor=20)
    # (3) l2 norm data and save the scaling factor
    l2f, snorm = to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # define image params
    lmbda = 0.14
    rvmin = 65  # dB
    vmax = 115  # dB
    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = 0.5

    W = np.ones_like(s)
    # First pass with unit weighting - same as ConvBPDN
    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() +eps
    # Caclulate sparse reconstruction
    rnorm = b.reconstruct().squeeze()
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)
    # Dictionary PSF
    Dout = b.D.squeeze().shape

    # Convert back from normalized
    s = from_l2_normed(snorm, l2f)
    x = from_l2_normed(xnorm, l2f)
    r = from_l2_normed(rnorm, l2f)

    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))
    r_log = 20 * np.log10(abs(r))

    match = match_histograms(x_log, r_log, multichannel=False)
    match_log = np.where(match <= np.min(match), 0, match)

    # Calculate a contour of the main reflector, the one that needs to be
    # deconvolved
    from scipy.signal import filtfilt, firwin

    b = firwin(25, 0.05)

    # Below this contour we will apply the speckle_weight
    # contour1 = np.argmax(gaussian_blur(r_log), axis=0) - 5
    #
    # contour1 = filtfilt(b, 1, contour1)
    #
    # W = np.ones(snorm.shape)
    # for i in range(W.shape[1]):
    #     W[int(contour1[i]):int(contour1[i] + 40), i] = speckle_weight
    #     # W[0:int(contour1[i]), i] = speckle_weight
    # contour1 = np.zeros(snorm.shape[1])
    # for i in range(len(contour1)):
    #     peaks, _ = find_peaks(abs(snorm[:,i]),height = 0.1)
    #     if len(peaks) == 0:
    #         contour1[i] = 0
    #     else:
    #     contour1[i] = peaks[0]
    # contour1 = filtfilt(b, 1, contour1)
    # contour1 = np.argmax(gaussian_blur(r_log), axis=0) - 5
    #
    # contour1 = filtfilt(b, 1, contour1)

    # W = np.ones(snorm.shape)
    # for i in range(W.shape[1]):
    #     W[int(contour1[i]):int(contour1[i] + 10), i] = speckle_weight
        # W[0:int(contour1[i]), i] = speckle_weight


    temp = gaussian_blur(x_log)
    W = np.where(x_log<=s_log.min(),1,speckle_weight)

    fig,ax = plt.subplots(2,4,figsize=(16,9))
    fig.suptitle('Pass 1 %d-%d' % (rvmin,vmax))
    ax[0,0].imshow(s_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,0].set_title('Reference')
    # ax[0,0].plot(contour1,linewidth=3)
    ax[1,0].plot(abs(snorm[:, index]))

    ax[0,1].imshow(r_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,1].set_title('Sparse rep Dx')

    ax[1,1].plot(abs(rnorm[:, index]))

    ax[0,2].imshow(x_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,2].set_title('Sparse vectors x')
    ax[1,2].plot(abs(xnorm[:, index]))

    ax[0, 3].imshow(match_log, 'gray', vmin=rvmin, vmax=vmax)
    ax[0, 3].set_title('Sparse vectors x(matched)')

    im = ax[1,3].imshow(W.squeeze())
    ax[1, 3].set_title('weighting matrix')
    divider = make_axes_locatable(ax[1,3])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.show()

    lmbda = 0.05
    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})


    b1 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Second time!
    x1norm = b1.solve().squeeze() +eps
    r1norm = b1.reconstruct().squeeze()
    x1norm = np.roll(x1norm, np.argmax(D), axis=0)

    x1 = from_l2_normed(x1norm, l2f)
    r1 = from_l2_normed(r1norm, l2f)

    x1_log = 20 * np.log10(abs(x1))
    r1_log = 20 * np.log10(abs(r1))


    match1 = match_histograms(x1_log, r1_log, multichannel=False)
    match1_log = np.where(match1 <= np.min(match1), 0, match1)
    x1_log_blur = gaussian_blur(x1_log, sigma=0.3)
    #
    # # %%
    # plot_images(['Reference', 'Sparse rep',
    #              'Sparse vectors', 'Sparse vectors (w/ blur)'],
    #             [s_log, r1_log, x1_log, x1_log_blur],
    #             rvmin, vmax,
    #             suptitle='Pass 2')
    # # %%
    # # plot_alines(['Reference', 'Sparse rep',
    # #              'Sparse vectors', 'Sparse vectors (w/ blur)'],
    # #             [s_log, r1_log, x1_log, x1_log_blur],
    # #             rvmin, vmax, suptitle='Pass 2')
    # # plt.show()

    fig,ax = plt.subplots(2,4,figsize=(16,9))
    fig.suptitle('Pass 2(weighted) %d-%d' % (rvmin,vmax))
    ax[0,0].imshow(s_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,0].set_title('Reference')
    # ax[0,0].plot(contour1,linewidth=3)
    ax[1,0].plot(abs(snorm[:, index]))

    ax[0,1].imshow(r1_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,1].set_title('Sparse rep Dx')

    ax[1,1].plot(abs(r1norm[:, index]))

    ax[0,2].imshow(x1_log,'gray',vmin = rvmin, vmax =vmax)
    ax[0,2].set_title('Sparse vectors x')
    ax[1,2].plot(abs(x1norm[:, index]))

    ax[0, 3].imshow(match1_log, 'gray', vmin=rvmin, vmax=vmax)
    ax[0, 3].set_title('Sparse vectors x(matched)')
    # ax[0, 3].plot(abs(xnorm[:, index]))

    im = ax[1,3].imshow(W.squeeze())
    ax[1, 3].set_title('weighting matrix')
    divider = make_axes_locatable(ax[1,3])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.show()

    # fig =plt.figure()
    # edges1 = feature.canny(x_log,sigma = 5,low_threshold=0.75, high_threshold=0.9)
    # edges1 = edges1.astype(float)
    # plt.imshow(edges1, cmap='gray')
    # plt.show()
    #
    # from scipy.signal import find_peaks
    #
    # M = np.ones(edges1.shape)
    # for i in range(edges1.shape[1]):
    #     temp = edges1[:, i]
    #     peaks, _ = find_peaks(temp)
    #     if len(peaks) < 2:
    #         continue
    #     else:
    #         M[peaks[0]:peaks[1],i] = 0.2
    #
    # plt.imshow(M)
    # plt.show()

    fig,ax = plt.subplots(2,2,figsize=(16,9))
    s_temp = pixel_norm(s_log)
    x_temp = pixel_norm(match1_log)

    # fig.suptitle('Young method')
    ax[0, 0].imshow(s_temp,'gray',vmax = 255,vmin = 160)
    ax[0, 0].set_title('reference norm_young')
    ax[0, 1].imshow(x_temp,'gray',vmax = 255,vmin = 160)
    ax[0, 1].set_title('final norm_young')

    ax[1, 0].imshow(imag2uint(s_log,rvmin,vmax),'gray',vmax = 255,vmin = 160)
    ax[1, 0].set_title('reference norm_rob')
    ax[1, 1].imshow(imag2uint(match1_log,rvmin,vmax),'gray',vmax = 255,vmin = 160)
    ax[1, 1].set_title('final norm_rob')

    plt.tight_layout()
    plt.show()

