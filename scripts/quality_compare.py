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
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from tabulate import tabulate


roi = {}
width, height = (20, 10)
roi['artifact'] = [[212, 142, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[212, 165, int(width * 1.2), int(height * 1.2)]]

# Module level constants
eps = 1e-14

if __name__ == '__main__':

    #Image processing and display paramaters
    speckle_weight = 0.1
    rvmin, vmax = 5, 55 #dB


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

    file_name,file_r,file_g = 'finger','finger(reference)','finger(gaussian)'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)

    s_r = processing.load_data(file_r, decimation_factor=20,data_only=True)
    s_g = processing.load_data(file_g, decimation_factor=20,data_only=True)

    lmbda = 0.03
    w_lmbda = 0.05

    x = processing.make_sparse_representation(s,D, lmbda,w_lmbda, speckle_weight)

    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))

    sr_log = 20 * np.log10(abs(s_r))
    sg_log = 20 * np.log10(abs(s_g))

    s_intensity = abs(s) ** 2
    x_intensity = abs(x) ** 2

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_r_1 = quality.ROI(*roi['homogeneous'][0], abs(s_r) ** 2)
    ho_g_1 = quality.ROI(*roi['homogeneous'][0], abs(s_g) ** 2)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)
    ar_r = quality.ROI(*roi['artifact'][0], abs(s_r) ** 2)
    ar_g = quality.ROI(*roi['artifact'][0], abs(s_g) ** 2)


    fig = plt.figure(figsize=(18, 13), constrained_layout = True)
    gs = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

    ax = fig.add_subplot(gs[0])
    ax.set_title('(a) no window', fontsize= 28)

    ax.imshow(sr_log, 'gray', aspect=sr_log.shape[1] / sr_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 50, roi['artifact'][0][1] - 60), textcoords='data', fontsize=30,
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

    textstr = r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_r_1, ar_r))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1])
    ax.set_title('(b) gaussian window', fontsize= 28)

    ax.imshow(sg_log, 'gray', aspect=sg_log.shape[1] / sg_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 50, roi['artifact'][0][1] - 60), textcoords='data', fontsize=30,
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

    textstr = r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_g_1, ar_g))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')


    ax = fig.add_subplot(gs[2])
    ax.set_title('(c) hann window', fontsize= 28)

    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 50, roi['artifact'][0][1] - 60), textcoords='data', fontsize=30,
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

    textstr = r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ar_s))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')


    ax = fig.add_subplot(gs[3])
    ax.set_title('(d) 𝜆 = %.2f \n $W$ = %.1f' % (lmbda, speckle_weight),fontsize = 28)

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 50, roi['artifact'][0][1] - 60), textcoords='data', fontsize=30,
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

    textstr = r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_x_1, ar_x))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=26,
            verticalalignment='top', fontname='Arial', color='white')

    plt.show()


