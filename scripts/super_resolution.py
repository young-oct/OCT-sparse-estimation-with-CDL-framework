# -*- coding: utf-8 -*-
# @Time    : 2021-05-25 4:46 p.m.
# @Author  : young wang
# @FileName: super_resolution.py
# @Software: PyCharm

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from tabulate import tabulate
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

# Module level constants
eps = 1e-14

def guassinfit(x,a,mean,sigma):
   return a*np.exp(-(x-mean)**2/(2*sigma**2))

def kernel(data):
   x = np.linspace(0, 330, 330)
   popt, _ = curve_fit(guassinfit, x, data)
   blur = guassinfit(x,popt[0],popt[1],popt[2]*0.75)
   # blur = blur/prox.norm_l2(blur)
   return blur, popt

if __name__ == '__main__':

    # Image processing and display paramaters
    speckle_weight = 0.1
    rvmin = 65  # dB
    vmax = 115  # dB

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
    file_name = 'ear'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)
    est_psf, _ = kernel(abs(D).squeeze())

    # l2 norm data and save the scaling factor
    _, snorm = processing.to_l2_normed(s)
    lmbda = 0.07
    w_lmbda = 0.02

    index = 325 # index A-line
    s_line = abs(snorm[:,index])

    x,x_line = processing.make_sparse_representation(s, D, lmbda, w_lmbda, speckle_weight,Line=True,index = index, Ear=True)

    b_line = np.convolve(x_line, est_psf, mode='same')

    temp = np.zeros((330, 512))

    for i in range(x.shape[1]):
        temp[:, i] = np.convolve(abs(x[:, i]), est_psf, mode='same')
        dis = np.argmax(abs(temp[:, i])) - np.argmax(temp[:, i])
        temp[:, i] = np.roll(temp[:, i], int(dis))

    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))
    b_log = 20 * np.log10(abs(temp))

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(ncols=3, nrows=2)

    aspect = s_log.shape[1] / s_log.shape[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(s_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin, interpolation='none')
    ax.set_axis_off()
    ax.set_title('(a) reference', fontname='Arial')
    ax.axvline(x=index, linewidth=2, color='orange', linestyle='--')

    ax = fig.add_subplot(gs[1, 0])

    ax.plot(s_line)
    ax.set_xlabel('axial depth [pixels]', fontname='Arial')
    ax.set_ylabel('normalized magnitude [a.u.]', fontname='Arial')
    axins = ax.inset_axes([0.02, 0.3, 0.37, 0.67])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.plot(s_line)

    axins.set_xlim(140, 180)
    axins.set_ylim(0, 0.3)

    textstr_an = 'distal''\n''sidelobe'
    textstr_po = 'proximal''\n''sidelobe'

    axins.annotate(textstr_an, xy=(147, 0.09), xycoords='data',
                   xytext=(148, 0.18), textcoords='data', fontsize=13,
                   color='red', fontname='Arial',
                   arrowprops=dict(facecolor='red', shrink=0.01),
                   horizontalalignment='center', verticalalignment='top')

    axins.annotate(textstr_po, xy=(168, 0.08), xycoords='data',
                   xytext=(170, 0.18), textcoords='data', fontsize=13,
                   color='red', fontname='Arial',
                   arrowprops=dict(facecolor='red', shrink=0.01),
                   horizontalalignment='center', verticalalignment='top')
    ax.indicate_inset_zoom(axins)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(x_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
    ax.axvline(x=index, linewidth=2, color='orange', linestyle='--')
    ax.set_title('(b) 𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda, speckle_weight))
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 1])

    ax.set_yticks([])
    ax.plot(x_line)
    axins = ax.inset_axes([0.02, 0.3, 0.37, 0.67])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.plot(x_line)

    axins.set_xlim(140, 180)
    axins.set_ylim(0, 0.3)
    ax.indicate_inset_zoom(axins)
    ax.set_xlabel('axial depth [pixels]')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(b_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
    ax.axvline(x=index, linewidth=2, color='orange', linestyle='--')
    ax.set_title('(c) 𝜆 = %.2f \n $\omega$ = %.1f(blurred)' % (lmbda, speckle_weight))
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 2])

    ax.set_yticks([])
    ax.plot(b_line)
    axins = ax.inset_axes([0.02, 0.3, 0.37, 0.67])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.plot(b_line)

    axins.set_xlim(140, 180)
    axins.set_ylim(0, 0.3)
    ax.indicate_inset_zoom(axins)
    ax.set_xlabel('axial depth [pixels]')


    plt.show()