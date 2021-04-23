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
from skimage.morphology import disk, square
from skimage.morphology import erosion, dilation
from skimage import feature
from PIL import Image
from scipy.signal import find_peaks


from skimage import feature
from scipy import ndimage, misc

# Module level constants
eps = 1e-14

def imag2uint(data, vmin, vmax):
    data=np.clip(data,vmin,vmax)
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals

def gaussian_blur(noisy, sigma=0.5):
    out = gaussian(noisy, sigma=sigma, output=None, mode='nearest', cval=0,
                   multichannel=None, preserve_range=False, truncate=4.0)
    return (out)

def load_data(dataset_name, decimation_factor):
    # define signal & dictionary path
    if dataset_name == 'finger':
        S_PATH = '../Data/nail'
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
                vmax, vmin, overlay_plots=None,
                suptitle=None):
    assert len(plot_titles) == len(plot_data), 'Length of plot_titles does not match length of plot_data'

    nplots = len(plot_titles)
    fig, axes = plt.subplots(1, nplots,figsize=(16, 9))
    if overlay_plots is None:
        overlay_plots = [None for i in range(len(plot_titles))]

    if suptitle is not None:
        plt.suptitle(suptitle+' %d-%d' %(vmin, vmax))
    for n, (ax, title, im, overlay_plot) in enumerate(zip(axes, plot_titles, plot_data, overlay_plots)):
        ax.set_title(title)
        ax.imshow(im, vmax=vmax, vmin=vmin, cmap='gray')
        ax.set_axis_off()
        if overlay_plot is not None:
            ax.plot(overlay_plot)
            ax.set_axis_off()
    plt.tight_layout()
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

def getWeight(speckle_weight):
    lmbda = 0.1
    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    x = from_l2_normed(xnorm, l2f)

    x_log = 20 * np.log10(abs(x))

    x_log = imag2uint(x_log,rvmin,vmax)

    #set thresdhold
    x_log = np.where(x_log <= rvmin,0,x_log)

    W = dilation(x_log,  disk(5))
    W = np.where(W > 0, speckle_weight,1)
    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W


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

    rvmin = 65  # dB
    vmax = 115  # dB
    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    lmbda = 0.03
    speckle_weight = 0.5

    W = getWeight(speckle_weight)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    xnorm = b.solve().squeeze()+eps
    rnorm = b.reconstruct().squeeze()
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    ## Convert back from normalized
    x = from_l2_normed(xnorm, l2f)
    r = from_l2_normed(rnorm, l2f)
    s = from_l2_normed(snorm, l2f)


    x_log = 20 * np.log10(abs(x))
    r_log = 20 * np.log10(abs(r))
    s_log = 20 * np.log10(abs(s))

    #normalize intensity
    x_log = imag2uint(x_log,rvmin,vmax)
    r_log = imag2uint(r_log, rvmin, vmax)
    s_log = imag2uint(s_log,rvmin,vmax)

    dmin, dmax = 0,255

    # plot_images(['Reference', 'weighting','Sparse rep',
    #              'Sparse vectors'],
    #             [s_log,W.squeeze(), r_log, x_log],
    #             dmax, dmin,
    #             suptitle='Pass 2')
    fig,ax = plt.subplots(1, 4, figsize=(16,9),constrained_layout=True)
    fig.suptitle('%d-%d' %(dmin, dmax))
    ax[0].imshow(s_log, 'gray',vmax = dmax, vmin = dmin)
    ax[0].set_axis_off()
    ax[0].set_title('reference')

    ax[2].imshow(r_log, 'gray',vmax = dmax, vmin = dmin)
    ax[2].set_axis_off()
    ax[2].set_title('Dx')

    ax[3].imshow(x_log, 'gray',vmax = dmax, vmin = dmin)
    ax[3].set_axis_off()
    ax[3].set_title('x')

    im = ax[1].imshow(W.squeeze())
    ax[1].set_title('weighting matrix')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1].set_axis_off()
    # plt.axis('off')

    plt.show()

    s_amp = np.zeros(s.shape[1])
    x_amp = np.zeros(s.shape[1])

    fig,ax = plt.subplots(1, 2,figsize=(16,9))
    for i in range(s.shape[1]):
        s_amp[i] = np.max(abs(snorm))
        x_amp[i] = np.max(abs(xnorm))
        
    ax[0].hist(s_amp,label='reference')
    ax[0].hist(x_amp,label='sparse')
    ax[0].legend()

    ax[1].plot(abs(snorm[:,200]),label='reference')
    ax[1].plot(abs(xnorm[:,200]),label='sparse')
    ax[1].legend()

    plt.show()








