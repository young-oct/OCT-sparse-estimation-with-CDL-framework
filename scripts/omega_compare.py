# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 8:48 p.m.
# @Author  : young wang
# @FileName: omega_compare.py
# @Software: PyCharm


'''this script generates images for the figure 3.1 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
various values of the weighting parameter omega'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
import matplotlib.patches as patches
from misc import processing,quality,annotation


# Module level constants
eps = 1e-14

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 16,
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
    _, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = np.linspace(0.1,1,5)
    lmbda = 0.05
    w_lmbda = 0.05

    index = 400 # index A-line
    s_line = abs(snorm[:,index])

    x_line = np.zeros((snorm.shape[0], len(speckle_weight)))
    sparse = np.zeros((snorm.shape[0], snorm.shape[1], len(speckle_weight)))
    sparsity = np.zeros(len(speckle_weight))

    #update opt to include W

    for i in range(len(speckle_weight)):

        x, line = processing.make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight[i],Line=True,
                                                        index = index , Ear=True)
        x_log = 20 * np.log10(abs(x))
        sparse[:,:,i] = x_log
        x_line[:, i] = line

    width, height = (115, 95)
    homogeneous = [[125, 120, width, height]]

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(ncols=len(speckle_weight) + 1, nrows=3)

    aspect = s_log.shape[1] / s_log.shape[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(s_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
    ax.set_axis_off()
    ax.set_title('reference')

    ax.axvline(x=index,linewidth=1, color='orange', linestyle='--')
    for k in range(len(homogeneous)):
        for j in annotation.get_homogeneous(*homogeneous[k]):
            ax.add_patch(j)

    ho_original = quality.ROI(*homogeneous[0], s_log)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ho_original, 'gray', aspect=ho_original.shape[1] / ho_original.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')
    ax.set_axis_off()
    ax.annotate('', xy=(72.5, 10), xycoords='data',
                xytext=(60, 5), textcoords='data',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    circ = patches.Circle((80, 55), 15, alpha=1, fill=False,edgecolor = 'red',
                          linestyle='--',transform=ax.transData)
    ax.add_patch(circ)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(s_line)
    ax.set_xlabel('axial depth [pixels]')
    ax.set_ylabel('normalized \nmagnitude [a.u.]', fontsize=14)
    ax.set_ylim(0, np.max(s_line)*1.1)

    temp = np.where(s_log <= rvmin,0,s_log)
    ba_s = quality.ROI(*homogeneous[0], temp)
    ar_s = quality.ROI(*homogeneous[0], temp)

    for i in range(len(speckle_weight)):

        aspect = sparse[:, :, i].shape[1]/sparse[:, :, i].shape[0]
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(sparse[:, :, i], 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
        ax.axvline(x=index, linewidth=1, color='orange', linestyle='--')

        textstr = '\n'.join((
            r'$𝜆$ = %.2f ' % (lmbda),
            r'$W$ = %.1f' % (speckle_weight[i])))
        ax.set_title(textstr)

        ax.set_axis_off()
        for k in range(len(homogeneous)):
            for j in annotation.get_homogeneous(*homogeneous[k]):
                ax.add_patch(j)

        ho_x = quality.ROI(*homogeneous[0], sparse[:, :, i])


        aspect = width / height
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(ho_x, 'gray', aspect=aspect, vmax=vmax,
                  vmin=rvmin,interpolation='none')
        ax.annotate('', xy=(72.5, 10), xycoords='data',
                    xytext=(60, 5), textcoords='data',
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        circ = patches.Circle((80, 55), 15, alpha=1, fill=False, edgecolor='red',
                              linestyle='--', transform=ax.transData)
        ax.add_patch(circ)

        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, i + 1])
        ax.plot(x_line[:, i])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(s_line)*1.1)

        ax.set_xlabel('axial depth [pixels]')
    plt.show()

