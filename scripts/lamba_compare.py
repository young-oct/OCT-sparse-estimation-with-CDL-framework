# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 7:00 p.m.
# @Author  : young wang
# @FileName: lamba_compare.py
# @Software: PyCharm

'''this script generates images for the figure 3 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
various values of the regularization parameter lambda'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.morphology import disk
from skimage.morphology import dilation
from misc import processing
from misc import quality
from functools import partial

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
            'font.size': 18,
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
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = 0.5
    lmbda = np.linspace(1e-2, 1e-1, 5)
    lmbda[1] = 0.03

    W = getWeight(0.1,speckle_weight)

    index = 256 # index A-line
    s_line = abs(snorm[:,index])

    x_line = np.zeros((snorm.shape[0], len(lmbda)))
    sparse = np.zeros((snorm.shape[0], snorm.shape[1], len(lmbda)))
    sparisty = np.zeros(len(lmbda))

    #update opt to include W
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    for i in range(len(lmbda)):
        b = cbpdn.ConvBPDN(D, snorm, lmbda[i], opt=opt_par, dimK=1, dimN=1)
        xnorm = b.solve().squeeze()

        #calculate sparsity
        sparisty[i] = (1-np.count_nonzero(xnorm) / xnorm.size)
        xnorm += eps

        xnorm = np.roll(xnorm, np.argmax(D), axis=0)

        x_line[:,i] = abs(xnorm[:,index])
        ## Convert back from normalized
        x = processing.from_l2_normed(xnorm, l2f)

        x_log = 20 * np.log10(abs(x))
        x_log = processing.imag2uint(x_log, rvmin, vmax)
        sparse[:,:,i] = x_log

    width, height = (100, 80)
    homogeneous = [[125, 120, width, height]]

    vmax, vmin = 255,0
    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(ncols=len(lmbda) + 1, nrows=3)

    aspect = s_log.shape[1] / s_log.shape[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(s_log, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    ax.set_title('reference', fontname='Arial')
    ax.axvline(x=index, ymin=0.6, ymax=1, linewidth=2, color='orange', linestyle='--')
    ax.axvline(x=index, ymin=0, ymax=0.6, linewidth=2, color='orange')

    ho_original = quality.ROI(*homogeneous[0], s_log)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ho_original, 'gray', aspect=ho_original.shape[1] / ho_original.shape[0], vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    ax.annotate('', xy=(72.5, 10), xycoords='data',
                xytext=(60, 5), textcoords='data',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    ax.annotate('', xy=(87.5, 55), xycoords='data',
                xytext=(92.5, 70), textcoords='data',
                arrowprops=dict(facecolor='red', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(s_line)

    for i in range(len(lmbda)):
        temp = sparse[:, :, i]
        aspect = temp.shape[1]/temp.shape[0]
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(temp, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
        ax.axvline(x=index, ymin=0.6, ymax=1, linewidth=2, color='orange', linestyle='--')
        ax.axvline(x=index, ymin=0, ymax=0.6, linewidth=2, color='orange')

        ax.set_title('𝜆 = %.3f \n $\omega$ = %.1f' % (lmbda[i], speckle_weight))
        ax.set_axis_off()

        ho_x = quality.ROI(*homogeneous[0], temp)
        #

        aspect = width / height
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(ho_x, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
        ax.annotate('', xy=(72.5, 10), xycoords='data',
                    xytext=(60, 5), textcoords='data',
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        ax.annotate('', xy=(87.5, 55), xycoords='data',
                    xytext=(92.5, 70), textcoords='data',
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, i + 1])
        ax.plot(x_line[:, i])
        ax.set_title('SF = %.3f'% sparisty[i])

    plt.show()

