# -*- coding: utf-8 -*-
# @Time    : 2021-05-14 4:28 p.m.
# @Author  : young wang
# @FileName: lambda_gCNR(onion).py
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

# Define ROIs
roi = {}
width, height = (20, 10)

roi['artifact'] = [[325, 120, int(width * 1.2), int(height * 1.2)]]
roi['background'] = [[400, 180, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[325, 140, int(width * 1.2), int(height * 1.2)],
                      [400, 155, int(width * 1.2), int(height * 1.2)]]


# Module level constants
eps = 1e-14
w_lmbda = 0.02

def lmbda_search(s,lmbda,speckle_weight):
    x = processing.make_sparse_representation(s,D, lmbda,w_lmbda,speckle_weight)

    s_intensity = abs(s)**2
    x_intensity = abs(x)**2

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    # calcuate image quality metrics

    #'gCNR ', 'H_1/A',
    gcnrh1a = quality.log_gCNR(ho_s_1, ar_s), quality.log_gCNR(ho_x_1, ar_x)

    #'gCNR', 'H_2/B',
    gcnrh2b = quality.log_gCNR(ho_s_2, ba_s), quality.log_gCNR(ho_x_2, ba_x)

    #'gCNR', 'H_1/H_2',
    gcnrh12 = quality.log_gCNR(ho_s_1, ho_s_2), quality.log_gCNR(ho_x_1, ho_x_2)

    #'gCNR', 'H_2/A',
    gcnrh2a = quality.log_gCNR(ho_s_2, ar_s), quality.log_gCNR(ho_x_2, ar_x)

    return (gcnrh1a,gcnrh2b,gcnrh12,gcnrh2a)

def value_plot(lmbda,value):

    gcnrh1a,gcnrh2b,gcnrh12,gcnrh2a = [],[],[],[]
    for i in range(len(value)):

        temp = value[i]
        gcnrh1a.append(temp[0][1])
        gcnrh2b.append(temp[1][1])
        gcnrh12.append(temp[2][1])
        gcnrh2a.append(temp[3][1])

    return lmbda[np.argmax(gcnrh2a)]

if __name__ == '__main__':

    #Image processing and display paramaters
    speckle_weight = 0.1
    rvmin, vmax = 5, 55 #dB

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 25,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    file_name = 'onion'
    # Load the example dataset
    s, D = processing.load_data(file_name, decimation_factor=20)
    lmbda = np.logspace(-4,-0.5,20)
    value = []
    for i in range(len(lmbda)):

        value.append(lmbda_search(s,lmbda[i],0.05))

    best = value_plot(lmbda,value)

    best = 0.01
    x = processing.make_sparse_representation(s,D, best,w_lmbda,speckle_weight)

    # Generate log intensity arrays
    s_intensity = abs(s) ** 2
    x_intensity = abs(x) ** 2

    s_log = 10 * np.log10(s_intensity)
    x_log = 10 * np.log10(x_intensity)

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)

    gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0], vmax=vmax,
              vmin=rvmin, interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] +50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 2*width, roi['artifact'][0][1] - 40 ), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0]+width, roi['background'][0][1] ), xycoords='data',
                xytext=(roi['background'][0][0] + 2*width, roi['background'][0][1] - 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_axis_off()
    ax.set_title('(a) reference', fontsize= 28)

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
        r'${SNR_{{H_2}/B}}$: %.1f $dB$' % (quality.SNR(ho_s_2, ba_s)),
        r'${C_{{H_2}/B}}$: %.1f $dB$' % (quality.Contrast(ho_s_2, ba_s)),
        r'${C_{{H_1}/{H_2}}}$: %.1f $dB$' % (quality.Contrast(ho_s_1, ho_s_2))))
    ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=22,
            verticalalignment='top', fontname='Arial', color='white')

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ar_s)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_2, ar_s)),
        r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(ho_s_2, ba_s)),
        r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ho_s_2))))
    ax.text(0.55, 0.98, textstr, transform=ax.transAxes, fontsize=22,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1])

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] +50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0]+width , roi['artifact'][0][1] ), xycoords='data',
                xytext=(roi['artifact'][0][0] + 2*width, roi['artifact'][0][1] - 40 ), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0]+width, roi['background'][0][1] ), xycoords='data',
                xytext=(roi['background'][0][0] + 2*width, roi['background'][0][1] - 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_title('(b) 𝜆 = %.2f \n $\omega$ = %.1f' % (best, speckle_weight),fontsize = 28)

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
        r'${SNR_{{H_2}/B}}: $%.1f $dB$''\t\t' % (quality.SNR(ho_x_2, ba_x)),
        r'${C_{{H_2}/B}}: $%.1f $dB$''\t\t' % (quality.Contrast(ho_x_2, ba_x)),
        r'${C_{{H_1}/{H_2}}}: $%.1f $dB$''\t\t' % (quality.Contrast(ho_x_1, ho_x_2))))
    ax.text(0.55, 0.98, textstr, transform=ax.transAxes, fontsize=22,
            verticalalignment='top', fontname='Arial', color='white')

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_1, ar_x)),
        r'${gCNR_{{H_2}/{A}}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_2, ar_x)),
        r'${gCNR_{{H_2}/B}}: $%.2f''\t\t' % (quality.log_gCNR(ho_x_2, ba_x)),
        r'${gCNR_{{H_1}/{H_2}}}: $%.2f' % (quality.log_gCNR(ho_x_1, ho_x_2))))
    ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=22,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[2])

    ax.text(0.5, 1.25, r'(c) $gCNR$ versus 𝜆 curves',
             horizontalalignment='center',
             fontsize=28,
             transform=ax.transAxes)
    reference = []

    for i in range(4):
        temp = value[0]
        reference.append(temp[i][0])

    gcnrh1a, gcnrh2b, gcnrh12, gcnrh2a = [], [], [], []
    for i in range(len(value)):
        temp = value[i]
        gcnrh1a.append(temp[0][1])
        gcnrh2b.append(temp[1][1])
        gcnrh12.append(temp[2][1])
        gcnrh2a.append(temp[3][1])

    ax.plot(lmbda, gcnrh1a, color='green', label=r'${gCNR_{{H_1}/{A}}}$')
    ax.axhline(reference[0], color='green', linestyle='--')

    ax.plot(lmbda, gcnrh2b, color='red', label=r'${gCNR_{{H_2}/{B}}}$')
    ax.axhline(reference[1], color='red', linestyle='--')

    ax.plot(lmbda, gcnrh12, color='orange', label=r'${gCNR_{{H_1}/{H_2}}}$')
    ax.axhline(reference[2], color='orange', linestyle='--')

    ax.plot(lmbda, gcnrh2a, color='purple', label=r'${gCNR_{{H_2}/{A}}}$')
    ax.axhline(reference[3], color='purple', linestyle='--')

    ax.set_ylabel(r'${gCNR}$',fontsize=20)

    ax.set_xlabel('𝜆 ')
    ax.set_xscale('log')
    ax.set_ylim(0.35, 1)
    ax.set_aspect(4.5)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0., fontsize = 18)

    plt.show()

    # table formant original then sparse
    table = [['SNR', 'H_2/B', quality.SNR(ho_s_2, ba_s), quality.SNR(ho_x_2, ba_x)],
             ['Contrast', 'H_2/B', quality.Contrast(ho_s_2, ba_s), quality.Contrast(ho_x_2, ba_x)],
             ['Contrast', 'H_1/H_2', quality.Contrast(ho_s_1, ho_s_2), quality.Contrast(ho_x_1, ho_x_2)],
             ['gCNR ', 'H_1/A', quality.log_gCNR(ho_s_1, ar_s), quality.log_gCNR(ho_x_1, ar_x)],
             ['gCNR', 'H_2/B', quality.log_gCNR(ho_s_2, ba_s), quality.log_gCNR(ho_x_2, ba_x)],
             ['gCNR', 'H_1/H_2', quality.log_gCNR(ho_s_1, ho_s_2), quality.log_gCNR(ho_x_1, ho_x_2)],
             ['gCNR', 'H_2/A', quality.log_gCNR(ho_s_2, ar_s), quality.log_gCNR(ho_x_2, ar_x)]]

    print(tabulate(table, headers=['IQA', 'Region', 'Reference image', 'Deconvolved image'],
                   tablefmt='fancy_grid', floatfmt='.2f', numalign='right'))