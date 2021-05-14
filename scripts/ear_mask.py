
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from skimage import filters
from skimage.morphology import disk
from misc import processing
import numpy as np
from sporco import prox
import pickle
from pathlib import Path
from scipy import ndimage
from skimage.morphology import disk,square,star,diamond,octagon
from sporco.admm import cbpdn
from skimage.morphology import dilation, erosion
from skimage import filters
from scipy.signal import find_peaks
from skimage import feature
from scipy.ndimage import gaussian_filter


# Module level constants
eps = 1e-14


def plot_images(plot_titles, image,
                vmin, vmax, suptitle=None, overlays=False):
    if overlays != True:
        assert len(plot_titles) == len(image), 'Length of plot_titles does not match length of plot_data'
        image = image
    else:
        assert len(plot_titles) + 1 == len(image)
        mask = image[-1]
        image.pop(-1)

    nplots = len(plot_titles)
    fig, axes = plt.subplots(2, (nplots + 1) // 2, figsize=(18, 13))

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes.flatten(), plot_titles, image)):

        if n == 4:
            ax.set_axis_off()
            ax = axes.flatten()[5]

        ax.set_title(title)

        if n != 1:


            if n == 3 and overlays == True:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')
                ax.contour(mask, [0.99], colors='orange', alpha=0.75, linestyles='dashed')
                axins.contour(mask, [0.99], colors='orange', alpha=0.75, linestyles='dashed')

            else:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')

        else:
            ax.plot(im)
            ax.set_xlabel('Axial depth [pixels]', fontname='Arial')
            ax.set_ylabel('Normalized \nmagnitude [a.u.]', fontname='Arial', fontsize=20)

            axins = ax.inset_axes([0.58, 0.2, 0.41, 0.6])
            axins.set_xticks([])
            axins.set_yticks([])
            axins.plot(im)
            axins.annotate('', xy=(156, 0.07), xycoords='data',
                           xytext=(150, 0.1), textcoords='data', fontsize=30,
                           color='red', fontname='Arial',
                           arrowprops=dict(facecolor='red', shrink=0.025),
                           horizontalalignment='left', verticalalignment='top')
            axins.annotate('', xy=(172, 0.09), xycoords='data',
                           xytext=(178, 0.12), textcoords='data', fontsize=30,
                           color='red', fontname='Arial',
                           arrowprops=dict(facecolor='red', shrink=0.025),
                           horizontalalignment='right', verticalalignment='top')

            axins.set_xlim(145, 180)
            axins.set_ylim(0, 0.15)
            ax.indicate_inset_zoom(axins)

    plt.tight_layout(pad=0.5)
    plt.show()


def getWeight(s, D, w_lmbda, speckle_weight, Paddging=True):
    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, w_lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    rvmin, vmax = 65, 115
    x = processing.from_l2_normed(xnorm, l2f)
    x_log = 10 * np.log10(abs(x) ** 2)
    x_log = processing.imag2uint(x_log, rvmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= rvmin, 0, x_log)

    W = dilation(x_log, disk(2))
    W = erosion(W, disk(2))
    W = np.where(W > 0, speckle_weight, 1)
    W = filters.median(W, square(7))
    height = 0.1

    if Paddging == True:
        pad = 15  #
        # find the bottom edge of the mask with Sobel edge filter

        temp = filters.sobel(W)

        pad_value = np.linspace(speckle_weight, 1, pad)

        for i in range(temp.shape[1]):
            peak, _ = find_peaks(temp[:, i], height=height)
            if len(peak) != 0:
                loc = peak[-1]
                if temp.shape[0] - loc >= pad:
                    W[loc:int(loc + pad), i] = pad_value
            else:
                W[:, i] = W[:, i]
    else:
        W = W

    return W

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 18,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )

    file_name = ['ear']

    lmbda = 0.04
    w_lmbda = 0.02
    speckle_weight = 0.1
    rvmin = 65  # dB
    vmax = 115  # dB

    s, D = processing.load_data(file_name[0], decimation_factor=20)
    # l2 norm data and save the scaling factor


    W = getWeight(s, D, w_lmbda, speckle_weight, Paddging=True)

    aspect = 512/330
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(16, 9),constrained_layout=True )

    s_log = 20*np.log10(abs(s))

    ax.imshow(s_log, 'gray',aspect=aspect,vmax=vmax, vmin=rvmin,interpolation='none')
    ax.contour(W, [0.99], colors='orange', alpha= 0.75 , linestyles='dashed')


    plt.show()

