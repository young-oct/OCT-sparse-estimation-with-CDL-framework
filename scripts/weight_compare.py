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
from skimage import filters
from skimage.morphology import disk
from skimage.morphology import dilation, erosion
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec


# Module level constants
eps = 1e-14


def plot_images(plot_titles, image,
                vmin, vmax,suptitle=None):
    assert len(plot_titles) == len(image), 'Length of plot_titles does not match length of plot_data'

    width, height = (17, 10)
    background = [[315, 30, width*5, height*5]]

    nplots = len(plot_titles)
    fig, axes = plt.subplots(1, nplots,figsize=(16,9),constrained_layout=True)

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes.flatten(), plot_titles, image)):
        ax.set_title(title)
        ax.set_axis_off()
        ax.imshow(im,aspect= im.shape[1]/im.shape[0], vmax=vmax, vmin=vmin, cmap='gray')

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

        for i in range(len(background)):
            for j in annotation.get_background(*background[i]):
                ax.add_patch(j)

        im = np.where(im <= rvmin,0,im)
        ba = quality.ROI(*background[0], im)
        textstr = r'${SF_{B}}$''\n'r'%.1f' % (quality.SF(ba))

        ax.text(0.8, 0.2, textstr, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', fontname='Arial', color='red')

    plt.show()



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

    # l2 norm data and save the scaling factor
    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = 0.1
    lmbda = 0.1

    b0 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    x0norm = b0.solve().squeeze() + eps
    r0norm = b0.reconstruct().squeeze()

    x0norm = np.roll(x0norm, np.argmax(D), axis=0)
    x0 = processing.from_l2_normed(x0norm, l2f)
    r0 = processing.from_l2_normed(r0norm, l2f)

    x0_log = 20 * np.log10(abs(x0))
    r0_log = 20 * np.log10(abs(r0))

    # update opt to include W
    # index = 290
    x1 = processing.make_sparse_representation(s, D, lmbda, speckle_weight)
    x1_log = 20 * np.log10(abs(x1))

    x1_median = filters.median(x1_log, disk(1)).squeeze()



    title = ['reference','sparse estimation \n 𝜆 = %.2f'% (lmbda),'𝜆 = %.2f'% (lmbda),
             '𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda,speckle_weight),
             '𝜆 = %.2f \n $\omega$ = %.1f(median)' % (lmbda, speckle_weight)]

    plot_images(title,[s_log,r0_log,x0_log,x1_log,x1_median],rvmin,vmax)
