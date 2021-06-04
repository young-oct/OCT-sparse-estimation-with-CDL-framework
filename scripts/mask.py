# -*- coding: utf-8 -*-
# @Time    : 2021-05-13 7:22 p.m.
# @Author  : young wang
# @FileName: mask.py
# @Software: PyCharm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from skimage import filters
from skimage.morphology import disk
from misc import processing

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

    file_name = ['ear', 'finger', 'nail', 'onion']
    title_name = ['middle ear', 'index finger (palmar view)', 'index finger (side view)', 'onion slice']

    original = []
    sparse = []
    W = []

    lmbda = 0.04
    w_lmbda = 0.05
    speckle_weight = 0.1
    rvmin, vmax = 5, 55 #dB


    for i in range(len(file_name)):
        # Load the example dataset

        Ear = False
        s, D = processing.load_data(file_name[i], decimation_factor=20)
        # l2 norm data and save the scaling factor
        l2f, snorm = processing.to_l2_normed(s)

        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

        # obtain weighting mask
        if file_name[i] == 'ear':
            Ear = True
        else:
            Ear = False
        x,mask = processing.make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight, Mask = True, Ear = Ear)

        x_log = 20 * np.log10(abs(x))
        s_log = 20 * np.log10(abs(s))

        original.append(s_log)
        sparse.append(x_log)
        W.append(mask)

    aspect = original[0].shape[1]/original[0].shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=True, figsize=(16, 9),constrained_layout=True )

    for i in range(len(file_name)):
        title = '\n'.join((title_name[i],'𝜆 = %.2f $\omega$ = %.1f' % (lmbda, speckle_weight)))

        ax[0, i].set_title(title,fontsize=20)
        ax[0, i].imshow(original[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin,interpolation='none')
        ax[0, i].contour(W[i], [0.99], colors='orange', alpha=0.75, linestyles='dashed')

        ax[1, i].imshow(sparse[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin,interpolation='none')
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    plt.show()
