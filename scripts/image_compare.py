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
from skimage.morphology import disk
from misc import processing
from skimage import filters

# Module level constants
eps = 1e-14


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

    lmbda = [0.1,0.04,0.02,0.03]
    speckle_weight = [0.1,0.1,0.3,0.1]
    rvmin = 65  # dB
    vmax = 115  # dB

    for i in range(len(file_name)):
        # Load the example dataset
        s, D = processing.load_data(file_name[i], decimation_factor=20)
        # l2 norm data and save the scaling factor
        l2f, snorm = processing.to_l2_normed(s)

        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

        # obtain weighting mask
        x = processing.make_sparse_representation(s, D, lmbda[i], speckle_weight[i])

        x_log = 20 * np.log10(abs(x))
        s_log = 20 * np.log10(abs(s))

        # apply median filter
        s_log = filters.median(s_log, disk(1))
        x_log = filters.median(x_log, disk(1))

        original.append(s_log)
        sparse.append(x_log)

    x_head = [300, 200, 240, 250]
    y_head = [110, 125, 170, 120]

    x_end = [350, 150, 190, 190]
    y_end = [90, 105, 150, 100]

    aspect = original[0].shape[1]/original[0].shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=True, figsize=(16, 9),constrained_layout=True )

    for i in range(len(file_name)):
        title = '\n'.join((title_name[i],'𝜆 = %.2f $\omega$ = %.1f' % (lmbda[i], speckle_weight[i])))

        ax[0, i].set_title(title,fontsize=20)
        ax[0, i].imshow(original[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin)
        ax[0, i].annotate('', xy=(x_head[i], y_head[i]), xycoords='data',
                          xytext=(x_end[i], y_end[i]), textcoords='data',
                          arrowprops=dict(facecolor='white', shrink=0.05),
                          horizontalalignment='right', verticalalignment='top',
                          )

        ax[1, i].imshow(sparse[i], 'gray',aspect=aspect,vmax=vmax, vmin=rvmin)
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    plt.show()
