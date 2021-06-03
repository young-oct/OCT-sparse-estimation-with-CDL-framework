# -*- coding: utf-8 -*-
# @Time    : 2021-05-25 3:05 p.m.
# @Author  : young wang
# @FileName: artifact_quality.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021-05-25 2:39 p.m.
# @Author  : young wang
# @FileName: weak_and_back.py
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
from tabulate import tabulate
from scipy.ndimage import median_filter


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
    file_name = 'finger'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)

    lmbda = 0.06
    w_lmbda = 0.02

    x = processing.make_sparse_representation(s, D, lmbda, w_lmbda, speckle_weight)

    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))

    s_intensity = abs(s) ** 2

    x_intensity = abs(x) ** 2

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)

    gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    axins = ax.inset_axes([300, 10,200, 100],transform=ax.transData)
    axins.imshow(s_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.set_xlim(212, 236)
    axins.set_ylim(142, 154)
    ax.indicate_inset_zoom(axins, edgecolor='green')

    axins = ax.inset_axes([10, 250, 200, 60],transform=ax.transData)
    axins.imshow(s_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.set_xlim(390, 414)
    axins.set_ylim(225, 237)
    ax.indicate_inset_zoom(axins, edgecolor='red')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1]), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0] + width, roi['artifact'][0][1]), xycoords='data',
                xytext=(roi['artifact'][0][0] + 2 * width, roi['artifact'][0][1] - 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0] + width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2 * width, roi['background'][0][1] + 40), textcoords='data',
                fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_axis_off()
    ax.set_title('(a) reference', fontsize=28)

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
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ar_s)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_2, ar_s))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1])


    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    axins = ax.inset_axes([300, 10,200, 100],transform=ax.transData)
    axins.imshow(x_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.set_xlim(212, 236)
    axins.set_ylim(142, 154)
    ax.indicate_inset_zoom(axins, edgecolor='green')

    axins = ax.inset_axes([10, 250, 200, 60],transform=ax.transData)
    axins.imshow(x_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.set_xlim(390, 414)
    axins.set_ylim(225, 237)
    ax.indicate_inset_zoom(axins, edgecolor='red')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1]), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0] + width, roi['artifact'][0][1]), xycoords='data',
                xytext=(roi['artifact'][0][0] + 2 * width, roi['artifact'][0][1] - 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0] + width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2 * width, roi['background'][0][1] + 40), textcoords='data',
                fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_title('(b) 𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda, speckle_weight), fontsize=28)

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

        r'${gCNR_{{H_1}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_1, ar_x,improvement=False)),
        r'${gCNR_{{H_2}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_2, ar_x,improvement=False))))

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')


    # ax = fig.add_subplot(gs[2])
    #
    # h_log = median_filter(x_log, size=(3, 3))
    # ax.imshow(h_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
    #           vmax=vmax, vmin=rvmin, interpolation='none')
    #
    # axins = ax.inset_axes([300, 10,200, 100],transform=ax.transData)
    # axins.imshow(h_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')
    #
    # axins.set_xlim(212, 236)
    # axins.set_ylim(142, 154)
    # ax.indicate_inset_zoom(axins, edgecolor='green')
    #
    # axins = ax.inset_axes([10, 250, 200, 60],transform=ax.transData)
    # axins.imshow(h_log, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')
    #
    # axins.set_xlim(390, 414)
    # axins.set_ylim(225, 237)
    # ax.indicate_inset_zoom(axins, edgecolor='red')
    #
    # text = r'${H_{1}}$'
    # ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
    #             xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
    #             color='white', fontname='Arial',
    #             arrowprops=dict(facecolor='white', shrink=0.025),
    #             horizontalalignment='right', verticalalignment='top')
    #
    # text = r'${H_{2}}$'
    # ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
    #             xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1]), textcoords='data', fontsize=30,
    #             color='white', fontname='Arial',
    #             arrowprops=dict(facecolor='white', shrink=0.025),
    #             horizontalalignment='right', verticalalignment='top')
    #
    # text = r'${A}$'
    # ax.annotate(text, xy=(roi['artifact'][0][0] + width, roi['artifact'][0][1]), xycoords='data',
    #             xytext=(roi['artifact'][0][0] + 2 * width, roi['artifact'][0][1] - 40), textcoords='data', fontsize=30,
    #             color='white', fontname='Arial',
    #             arrowprops=dict(facecolor='white', shrink=0.025),
    #             horizontalalignment='left', verticalalignment='top')
    #
    # text = r'${B}$'
    # ax.annotate(text, xy=(roi['background'][0][0] + width, roi['background'][0][1] + height), xycoords='data',
    #             xytext=(roi['background'][0][0] + 2 * width, roi['background'][0][1] + 40), textcoords='data',
    #             fontsize=30,
    #             color='white', fontname='Arial',
    #             arrowprops=dict(facecolor='white', shrink=0.025),
    #             horizontalalignment='left', verticalalignment='top')
    #
    # ax.set_title('(c) 𝜆 = %.2f \n $\omega$ = %.1f (median filter)' % (lmbda, speckle_weight), fontsize=28)
    #
    # ax.set_axis_off()
    # for i in range(len(roi['artifact'])):
    #     for j in annotation.get_artifact(*roi['artifact'][i]):
    #         ax.add_patch(j)
    # for i in range(len(roi['background'])):
    #     for j in annotation.get_background(*roi['background'][i]):
    #         ax.add_patch(j)
    # for i in range(len(roi['homogeneous'])):
    #     for j in annotation.get_homogeneous(*roi['homogeneous'][i]):
    #         ax.add_patch(j)
    #
    # textstr = '\n'.join((
    #     r'${gCNR_{{H_1}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_1, ar_x,improvement=True)),
    #     r'${gCNR_{{H_2}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_2, ar_x,improvement=True))))
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
    #         verticalalignment='top', fontname='Arial', color='white')

    plt.show()
