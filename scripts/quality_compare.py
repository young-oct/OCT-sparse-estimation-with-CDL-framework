# -*- coding: utf-8 -*-
# @Time    : 2021-05-10 10:05 p.m.
# @Author  : young wang
# @FileName: quality_compare.py
# @Software: PyCharm


"""this script generates images for the figure 5 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
optimal values of the weighting parameter and lambda"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.morphology import disk
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from skimage import filters
from tabulate import tabulate

bin_n = 200
# Define ROIs
roi = {}
width, height = (20, 10)
roi['artifact'] = [[212, 142, int(width * 1.2), int(height * 1.2)]]
roi['background'] = [[390, 247, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[212, 165, int(width * 1.2), int(height * 1.2)],
                      [390, 225, int(width * 1.2), int(height * 1.2)]]


# Module level constants
eps = 1e-14

if __name__ == '__main__':

    #Image processing and display paramaters
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
    file_name = 'finger'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)

    lmbda = 0.04

    x = processing.make_sparse_representation(s,D, lmbda, speckle_weight)

    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    s_log = filters.median(s_log, disk(1))
    x_log = 20 * np.log10(abs(x))
    x_log = filters.median(x_log, disk(1))

    s_intensity = abs(s) ** 2
    s_intensity = filters.median(s_intensity, disk(1))

    x_intensity = abs(x) ** 2
    x_intensity = filters.median(x_intensity, disk(1))

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0], vmax=vmax, vmin=rvmin)

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    ax.set_axis_off()
    ax.set_title('reference')

    ax.set_aspect(s_log.shape[1] / s_log.shape[0])

    for i in range(len(roi['artifact'])):
        for j in annotation.get_artifact(*roi['artifact'][i]):
            ax.add_patch(j)
    for i in range(len(roi['background'])):
        for j in annotation.get_background(*roi['background'][i]):
            ax.add_patch(j)
    for i in range(len(roi['homogeneous'])):
        for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
            ax.add_patch(j)

    textstr = '\n'.join((
        r'${SNR_{{H_2}/B}}$''\n'
        r'%.1f dB' % (quality.SNR(ho_s_2, ba_s)),
        r'${C_{{H_2}/B}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_s_2, ba_s)),
        r'${C_{{H_1}/{H_2}}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_s_1, ho_s_2)),
        r'${gCNR_{{H_1}/{A}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_s_1, ar_s, N=bin_n)),
        r'${gCNR_{{H_2}/{A}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_s_2, ar_s, N=bin_n)),
        r'${gCNR_{{H_2}/B}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_s_2, ba_s, N=bin_n)),
        r'${gCNR_{{H_1}/{H_2}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_s_1, ho_s_2, N=bin_n))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1])

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0], vmax=vmax, vmin=rvmin)

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    ax.set_title('𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda, speckle_weight))

    ax.set_axis_off()
    for i in range(len(roi['artifact'])):
        for j in annotation.get_artifact(*roi['artifact'][i]):
            ax.add_patch(j)
    for i in range(len(roi['background'])):
        for j in annotation.get_background(*roi['background'][i]):
            ax.add_patch(j)
    for i in range(len(roi['homogeneous'])):
        for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
            ax.add_patch(j)

    textstr = '\n'.join((
        r'${SNR_{{H_2}/B}}$''\n'
        r'%.1f dB' % (quality.SNR(ho_x_2, ba_x)),
        r'${C_{{H_2}/B}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_x_2, ba_x)),
        r'${C_{{H_1}/{H_2}}}$''\n'
        r'%.1f dB' % (quality.Contrast(ho_x_1, ho_x_2)),
        r'${gCNR_{{H_1}/{A}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_x_1, ar_x, N=bin_n)),
        r'${gCNR_{{H_2}/{A}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_x_2, ar_x, N=bin_n)),
        r'${gCNR_{{H_2}/B}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_x_2, ba_x, N=bin_n)),
        r'${gCNR_{{H_1}/{H_2}}}$''\n'
        r'%.2f ' % (quality.gCNR(ho_x_1, ho_x_2, N=bin_n))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', fontname='Arial', color='white')


    plt.tight_layout()
    plt.show()

    # table formant original then sparse
    table = [['SNR', 'H_2/B', quality.SNR(ho_s_2, ba_s), quality.SNR(ho_x_2, ba_x)],
             ['Contrast', 'H_2/B', quality.Contrast(ho_s_2, ar_s), quality.Contrast(ho_x_2, ar_x)],
             ['Contrast', 'H_1/H_2', quality.Contrast(ho_s_1, ho_s_2), quality.Contrast(ho_x_1, ho_x_2)],
             ['gCNR ', 'H_1/A', quality.gCNR(ho_s_1, ar_s, N=bin_n), quality.gCNR(ho_x_1, ar_x, N=bin_n)],
             ['gCNR', 'H_2/B', quality.gCNR(ho_s_2, ba_s, N=bin_n), quality.gCNR(ho_x_2, ba_x, N=bin_n)],
             ['gCNR', 'H_1/H_2', quality.gCNR(ho_s_1, ho_s_2, N=bin_n), quality.gCNR(ho_x_1, ho_x_2, N=bin_n)],
             ['gCNR', 'H_2/A', quality.gCNR(ho_s_2, ar_s, N=bin_n), quality.gCNR(ho_x_2, ar_x, N=bin_n)]]

    print(tabulate(table, headers=['IQA', 'Region', 'Reference image', 'Deconvolved image'],
                   tablefmt='fancy_grid', floatfmt='.2f', numalign='right'))

    # from PIL import Image
    # out = processing.imag2uint(s_log,rvmin,vmax)
    # out = Image.fromarray(out)
    # out.save('/Users/youngwang/Desktop/original.bmp')
    #
    # out = processing.imag2uint(x_log,rvmin,vmax)
    # out = Image.fromarray(out)
    # out.save('/Users/youngwang/Desktop/sparse.bmp')

