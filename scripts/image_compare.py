# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:49 p.m.
# @Author  : young wang
# @FileName: image_compare.py
# @Software: PyCharm

'''this script generates images for the figure 2 as seen in
the paper. From left to right: OCT images of a middle ear,
index finger (palmar view), index finger (side view),
 and onion slice. The top row images are obtained from
 conventional OCT image processing produced from DFT
 of the histogram mean subtraction. Both sets of
 images are mapped onto 256-greyscale intensities.
 The reference images are displayed from 0-255 in
  grayscale values, the sparse images are displayed in the range of 0-255.
  Same display scheme is used for other figures in this paper'''


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.morphology import disk
from skimage.morphology import dilation
from misc import processing

# Module level constants
eps = 1e-14

def getWeight(lmbda,speckle_weight):
    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    x = processing.from_l2_normed(xnorm, l2f)

    x_log = 20 * np.log10(abs(x))
    x_log = processing.imag2uint(x_log,rvmin,vmax)

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

    file_name = ['ear', 'finger', 'nail', 'onion']

    original = []
    sparse = []
    for i in range(len(file_name)):
        # Load the example dataset
        s, D = processing.load_data(file_name[i], decimation_factor=20)
        # l2 norm data and save the scaling factor
        l2f, snorm = processing.to_l2_normed(s)

        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

        rvmin = 65  # dB
        vmax = 115  # dB
        # Weigth factor to apply to the fidelity (l2) term in the cost function
        # in regions segmented as containing speckle
        lmbda = 0.03
        speckle_weight = 0.5

        # obtain weighting mask
        W = getWeight(0.1, speckle_weight)
        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

        b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
        xnorm = b.solve().squeeze() + eps
        rnorm = b.reconstruct().squeeze()
        xnorm = np.roll(xnorm, np.argmax(D), axis=0)

        ## Convert back from normalized
        x = processing.from_l2_normed(xnorm, l2f)

        x_log = 20 * np.log10(abs(x))
        s_log = 20 * np.log10(abs(s))

        # normalize intensity
        x_log = processing.imag2uint(x_log, rvmin, vmax)
        s_log = processing.imag2uint(s_log, rvmin, vmax)

        original.append(s_log)
        sparse.append(x_log)

    x_head = [300, 200, 240, 250]
    y_head = [110, 125, 170, 120]

    x_end = [350, 150, 190, 190]
    y_end = [90, 105, 150, 100]

    vmax, vmin = 255,0

    aspect = original[0].shape[1]/original[0].shape[1]
    fig, ax = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=True, figsize=(16, 9))

    for i in range(len(file_name)):
        ax[0, i].set_title(file_name[i],fontsize=20)
        ax[0, i].imshow(original[i], 'gray',aspect=aspect,vmax=vmax, vmin=vmin)
        ax[0, i].annotate('', xy=(x_head[i], y_head[i]), xycoords='data',
                          xytext=(x_end[i], y_end[i]), textcoords='data',
                          arrowprops=dict(facecolor='white', shrink=0.05),
                          horizontalalignment='right', verticalalignment='top',
                          )

        if i != 0:
            axins = inset_axes(ax[0, i],
                               width="100%",  # width = 5% of parent_bbox width
                               height="100%",  # height : 50%
                               loc='upper right',
                               borderpad=0,
                               bbox_to_anchor=(0.60, 0.77, 0.38, 0.21),
                               bbox_transform=ax[0, i].transAxes
                               )
            axins.hist(np.ravel(original[i]), bins=18, density=True, log=True)
            axins.set_xlabel('intensity', color='white', fontname='Arial')
            axins.yaxis.set_visible(False)
            axins.set_xlabel('intensity', color='white', fontname='Arial')
            axins.tick_params(color='white', labelcolor='white')

            axins = inset_axes(ax[1, i],
                               width="100%",  # width = 5% of parent_bbox width
                               height="100%",  # height : 50%
                               loc='upper right',
                               borderpad=0,
                               bbox_to_anchor=(0.60, 0.77, 0.38, 0.21),
                               bbox_transform=ax[1, i].transAxes
                               )

            axins.hist(np.ravel(sparse[i]), bins=18, density=True, log=True)
            axins.set_xlabel('intensity', color='white', fontname='Arial')
            axins.yaxis.set_visible(False)

            axins.set_xlabel('intensity', color='white', fontname='Arial')
            axins.tick_params(color='white', labelcolor='white')

        ax[1, i].imshow(sparse[i], 'gray',aspect=aspect,vmax=vmax, vmin=vmin)
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, left= 0.005, right=0.995, hspace=-0.575,
                        wspace= 0.05)
    plt.show()
