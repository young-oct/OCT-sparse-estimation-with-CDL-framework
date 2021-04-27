# -*- coding: utf-8 -*-
# @Time    : 2021-04-27 7:29 p.m.
# @Author  : young wang
# @FileName: weight_compare.py
# @Software: PyCharm

"""this script generates images for the figure 2 as seen in
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


# Module level constants
eps = 1e-14

def getWeight(lmbda, speckle_weight):
    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized

    x = processing.from_l2_normed(xnorm, l2f)
    x_log = 20 * np.log10(abs(x))
    x_log = processing.imag2uint(x_log, rvmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= rvmin, 0, x_log)

    W = dilation(x_log, disk(5))
    W = erosion(W, disk(5))

    W = np.where(W > 0, speckle_weight, 1)
    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W


def plot_images(plot_titles, image,line,
                vmin, vmax,suptitle=None):
    assert len(plot_titles) == len(image), 'Length of plot_titles does not match length of plot_data'

    nplots = len(plot_titles)
    fig, axes = plt.subplots(2, nplots,figsize=(16,9))

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes[0,:], plot_titles, image)):
        ax.set_title(title)
        ax.set_axis_off()
        ax.imshow(im,aspect= im.shape[1]/im.shape[0], vmax=vmax, vmin=vmin, cmap='gray')
        ax.axvline(x=index,linewidth=1, color='orange', linestyle='--')

        ax.annotate('', xy=(200, 120), xycoords='data',
                    xytext=(180, 100), textcoords='data', fontsize=30,
                    color='white', fontname='Arial',
                    arrowprops=dict(facecolor='white', shrink=0.025),
                    horizontalalignment='right', verticalalignment='top')

        ax.annotate('', xy=(350, 295), xycoords='data',
                    xytext=(380, 275), textcoords='data', fontsize=30,
                    color='white', fontname='Arial',
                    arrowprops=dict(facecolor='white', shrink=0.025),
                    horizontalalignment='left', verticalalignment='top')

        ax.annotate('', xy=(140, 270), xycoords='data',
                    xytext=(170, 290), textcoords='data', fontsize=30,
                    color='red', fontname='Arial',
                    arrowprops=dict(facecolor='red', shrink=0.025),
                    horizontalalignment='right', verticalalignment='top')

        ax.annotate('', xy=(50, 90), xycoords='data',
                    xytext=(70, 110), textcoords='data', fontsize=30,
                    color='red', fontname='Arial',
                    arrowprops=dict(facecolor='red', shrink=0.025),
                    horizontalalignment='right', verticalalignment='top')

    for n,(ax,l) in enumerate(zip(axes[1,:],line)):
        ax.plot(l)
        ax.set_ylim(0, 0.52)
        ax.set_xlabel('axial depth [pixels]', fontname='Arial')
        if n != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('normalized intensity [a.u.]', fontname='Arial')
    plt.tight_layout()
    plt.show()

def get_line(index,data):
    aline = []
    for i in range(len(data)):
        aline.append(abs(data[i][:,index]))
    return aline

if __name__ == '__main__':

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
    file_name = ['ear']
    # Load the example dataset
    s, D = processing.load_data(file_name[0], decimation_factor=20)

    rvmin = 65  # dB
    vmax = 115  # dB

    s_log = 20 * np.log10(abs(s))
    s_log = processing.imag2uint(s_log, rvmin, vmax)

    # l2 norm data and save the scaling factor
    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = 0.3
    lmbda = 0.1

    b0 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    x0norm = b0.solve().squeeze() + eps
    r0norm = b0.reconstruct().squeeze()

    x0norm = np.roll(x0norm, np.argmax(D), axis=0)
    x0 = processing.from_l2_normed(x0norm, l2f)
    r0 = processing.from_l2_normed(r0norm, l2f)

    x0_log = 20 * np.log10(abs(x0))
    r0_log = 20 * np.log10(abs(r0))
    s_log = 20 * np.log10(abs(s))

    # normalize intensity
    x0_log = processing.imag2uint(x0_log, rvmin, vmax)
    r0_log = processing.imag2uint(r0_log, rvmin, vmax)
    s_log = processing.imag2uint(s_log, rvmin, vmax)

    # update opt to include W
    W = np.roll(getWeight(0.05,speckle_weight), np.argmax(D), axis=0)
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})


    b1 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    x1norm = b1.solve().squeeze() + eps

    # calculate sparsity
    x1norm = np.roll(x1norm, np.argmax(D), axis=0)

    ## Convert back from normalized
    x1 = processing.from_l2_normed(x1norm, l2f)

    x1_log = 20 * np.log10(abs(x1))
    x1_log = processing.imag2uint(x1_log, rvmin, vmax)

    vmax,vmin = 255,0

    data = [snorm,r0norm,x0norm,x1norm]
    index = 290
    aline = get_line(index,data)

    title = ['reference','sparse estimation \n 𝜆 = %.1f'% (lmbda),'𝜆 = %.2f'% (lmbda),
             '𝜆 = %.1f \n $\omega$ = %.1f' % (lmbda,speckle_weight)]

    plot_images(title,[s_log,r0_log,x0_log,x1_log],
                aline,vmin,vmax)
