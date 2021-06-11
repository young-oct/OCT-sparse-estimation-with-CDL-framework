# -*- coding: utf-8 -*-
# @Time    : 2021-04-27 7:29 p.m.
# @Author  : young wang
# @FileName: weight_compare.py
# @Software: PyCharm

"""Figure 2. Example of applying CBPDN-DL to an
OCT image of a middle ear with a mask"""
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
            ax.annotate('', xy=(200, 125), xycoords='data',
                        xytext=(175, 115), textcoords='data', fontsize=30,
                        color='white', fontname='Arial',
                        arrowprops=dict(facecolor='red', shrink=0.025),
                        horizontalalignment='right', verticalalignment='top')

            ax.annotate('', xy=(350, 295), xycoords='data',
                        xytext=(380, 280), textcoords='data', fontsize=30,
                        color='white', fontname='Arial',
                        arrowprops=dict(facecolor='red', shrink=0.025),
                        horizontalalignment='left', verticalalignment='top')

            ax.annotate('', xy=(60, 105), xycoords='data',
                        xytext=(80, 125), textcoords='data', fontsize=30,
                        color='red', fontname='Arial',
                        arrowprops=dict(facecolor='white', shrink=0.025),
                        horizontalalignment='right', verticalalignment='top')

            ax.annotate('', xy=(155, 270), xycoords='data',
                        xytext=(180, 290), textcoords='data', fontsize=30,
                        color='red', fontname='Arial',
                        arrowprops=dict(facecolor='white', shrink=0.025),
                        horizontalalignment='right', verticalalignment='top')

            ax.set_axis_off()

            scale = 2.5
            axins = ax.inset_axes([0.58, 0.45, 80 / 512 * scale, 87 / 330 * scale])
            axins.imshow(im, cmap='gray', vmax=vmax, vmin=vmin, interpolation='none')
            axins.set_xticklabels('')
            axins.set_yticklabels('')

            axins.set_xlim(22, 102)
            axins.set_ylim(110, 35)
            ax.indicate_inset_zoom(axins, edgecolor='white')

            if n == 3 and overlays == True:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')
                ax.contour(mask, [0.99], colors='orange', alpha=0.75, linestyles='dashed')
                axins.contour(mask, [0.99], colors='orange', alpha=0.75, linestyles='dashed')

            else:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')

        else:
            ax.plot(im)
            ax.set_xlabel('axial depth [pixels]')
            ax.set_ylabel('normalized \nmagnitude [a.u.]', fontsize=20)

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
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )
    file_name = ['ear']
    # Load the example dataset
    s, D = processing.load_data(file_name[0], decimation_factor=20)

    rvmin, vmax = 5, 55 #dB


    s_log = 20 * np.log10(abs(s))

    # l2 norm data and save the scaling factor
    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = 0.1
    lmbda = 0.05
    w_lmbda = 0.05

    b0 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    x0norm = b0.solve().squeeze() + eps
    r0norm = b0.reconstruct().squeeze()

    x0norm = np.roll(x0norm, np.argmax(D), axis=0)
    x0 = processing.from_l2_normed(x0norm, l2f)
    r0 = processing.from_l2_normed(r0norm, l2f)

    x0_log = 20 * np.log10(abs(x0))
    r0_log = 20 * np.log10(abs(r0))

    # update opt to include W
    x1, W = processing.make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight, Mask=True, Ear=True)
    x1_log = 20 * np.log10(abs(x1))

    title = [r'(a) reference',
             r'(b) magnitude of learned PSF $d(z)$',
             '\n'.join((r'(c) sparse estimate image', r'$𝜆$ = %.2f' % (lmbda))),
             '\n'.join((r'(d) sparse vector image', r'wo/weighting ($𝜆$ = %.2f)' % (lmbda))),
             '\n'.join((r'(e) sparse vector image', r'w/weighting ($𝜆$ = %.2f,$W$ = %.1f)' %  (lmbda, speckle_weight)))]

    plot_images(title, [s_log, abs(D), r0_log, x0_log, x1_log, W], rvmin, vmax, overlays=True)
