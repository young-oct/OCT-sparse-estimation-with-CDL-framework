# -*- coding: utf-8 -*-
# @Time    : 2021-11-08 8:13 a.m.
# @Author  : young wang
# @FileName: despeckling.py
# @Software: PyCharm



"""this script generates images for the figure 5 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
optimal values of the weighting parameter and lambda"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from scipy.ndimage import median_filter

from tabulate import tabulate
from matplotlib.ticker import (MultipleLocator)
import matplotlib.ticker
from skimage.restoration import denoise_bilateral
from scipy import ndimage, misc

# Define ROIs
roi = {}
width, height = (20, 10)
roi['artifact'] = [[212, 142, int(width * 1.2), int(height * 1.2)]]
roi['background'] = [[390, 260, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[212, 165, int(width * 1.2), int(height * 1.2)],
                      [390, 230, int(width * 1.2), int(height * 1.2)]]


# Module level constants
eps = 1e-14
bins = 32
w_lmbda = 0.05

def imagconvert(data, vmin, vmax):
    data = np.clip(data, vmin, vmax)
    pixel_vals = np.around(255 * (data - vmin) / (vmax - vmin), 0)
    return pixel_vals


def log_gCNR(region_h, region_b):
    assert np.size(region_h) == np.size(region_b), \
        'size of image patch'

    N = 256

    min_val, max_val = 0, 255

    h_hist, edge = np.histogram(np.ravel(region_h), bins=N, range=(min_val, max_val), density=True)
    h_hist = h_hist * np.diff(edge)
    b_hist, edge = np.histogram(np.ravel(region_b), bins=N, range=(min_val, max_val), density=True)
    b_hist = b_hist * np.diff(edge)

    ovl = 0

    for i in range(0,N):

        ovl += min(h_hist[i], b_hist[i])

    return 1 - ovl


def anote(ax,s):
    legend_font = 14

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0], roi['artifact'][0][1]), xycoords='data',
                xytext=(roi['artifact'][0][0] - 100, roi['artifact'][0][1] - 45), textcoords='data',
                fontsize=legend_font,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data',
                fontsize=legend_font,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 60, roi['homogeneous'][1][1]+10), textcoords='data',
                fontsize=legend_font,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0] + width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2 * width, roi['background'][0][1] + 40), textcoords='data',
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_axis_off()

    for i in range(len(roi['artifact'])):
        for j in annotation.get_artifact(*roi['artifact'][i]):
            ax.add_patch(j)

    for i in range(len(roi['homogeneous'])):
        for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
            ax.add_patch(j)

    for i in range(len(roi['background'])):
        for j in annotation.get_background(*roi['background'][i]):
            ax.add_patch(j)

    h1 = quality.ROI(*roi['homogeneous'][0], s)
    h2 = quality.ROI(*roi['homogeneous'][1], s)
    ba = quality.ROI(*roi['background'][0], s)
    ar = quality.ROI(*roi['artifact'][0], s)

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (log_gCNR(h1, ar)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (log_gCNR(h2, ar)),
        r'${gCNR_{{H_2}/B}}$: %.2f' % (log_gCNR(h2, ba)),
        r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (log_gCNR(h1, h2))))
    ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=legend_font,weight='bold',
            verticalalignment='top', fontname='Arial', color='white')
    return ax

if __name__ == '__main__':

    #Image processing and display paramaters
    speckle_weight = 0.1
    rvmin, vmax = 5, 55 #dB

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
    file_name = 'finger'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)

    best = 0.03

    x = processing.make_sparse_representation(s,D, best,w_lmbda,speckle_weight)

    # # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))

    s_log = imagconvert(s_log, rvmin, vmax)
    x_log = imagconvert(x_log, rvmin, vmax)

    x_median = ndimage.median_filter(x_log, size=3)
    x_bilateral = denoise_bilateral(x_log, sigma_color=None, sigma_spatial=15,
                                    multichannel=False)

    x_comb = denoise_bilateral(x_median, sigma_color=None, sigma_spatial=15,
                                    multichannel=False)

    images = [s_log,x_log,x_median,x_bilateral, x_comb]

    methods = ['(a) reference', '(b) deconvolved', '(c) median', '(d) bilateral', '(e) median+bilateral']
    fig,axs = plt.subplots(ncols=5, nrows=1,figsize=(16, 9), constrained_layout=True)

    for ax, title, image in zip(axs.flat, methods,images) :
        ax.imshow(image,'gray',aspect=s_log.shape[1] / s_log.shape[0], interpolation='none')
        anote(ax, image)

        ax.set_axis_off()
        ax.set_title(str(title))

    fig.savefig('../Images/despeckling.pdf',
                dpi = 600,
                transparent=True,format = 'pdf')

    plt.show()
