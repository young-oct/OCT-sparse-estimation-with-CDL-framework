# -*- coding: utf-8 -*-
# @Time    : 2021-05-07 2:54 p.m.
# @Author  : young wang
# @FileName: lambda_gCNR.py
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
def anote(ax,s,median_flag =False):
    legend_font = 15

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

    if median_flag == True:

        textstr = '\n'.join((
            r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(h1, ar,improvement=True)),
            r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(h2, ar,improvement=True)),
            r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(h2, ba,improvement=True)),
            r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(h1, h2,improvement=True))))
        ax.text(0.55, 0.98, textstr, transform=ax.transAxes, fontsize=legend_font,
                verticalalignment='top', fontname='Arial', color='white')

    else:

        textstr = '\n'.join((
            r'${SNR_{{H_2}/B}}$: %.1f $dB$' % (quality.SNR(h2, ba)),
            r'${C_{{H_2}/B}}$: %.1f $dB$' % (quality.Contrast(h2, ba)),
            r'${C_{{H_1}/{H_2}}}$: %.1f $dB$' % (quality.Contrast(h1, h2))))
        ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=legend_font,
                verticalalignment='top', fontname='Arial', color='white')

        textstr = '\n'.join((
            r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(h1, ar)),
            r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(h2, ar)),
            r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(h2, ba)),
            r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(h1, h2))))
        ax.text(0.55, 0.98, textstr, transform=ax.transAxes, fontsize=legend_font,
                verticalalignment='top', fontname='Arial', color='white')
    return ax

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

    fig,ax = plt.subplots(1,1, figsize=(16,9))
    ax.set_title(r'Generalized $CNR$ versus $𝜆$')
    reference = []

    for i in range(4):
        temp = value[0]
        reference.append(temp[i][0])

    gcnrh1a,gcnrh2b,gcnrh12,gcnrh2a = [],[],[],[]
    for i in range(len(value)):

        temp = value[i]
        gcnrh1a.append(temp[0][1])
        gcnrh2b.append(temp[1][1])
        gcnrh12.append(temp[2][1])
        gcnrh2a.append(temp[3][1])

    ax.plot(lmbda, gcnrh1a,color='green', label = r'${gCNR_{{H_1}/{A}}}$')
    ax.axhline(reference[0],color='green',linestyle = '--')

    ax.plot(lmbda, gcnrh2b,color='red',label = r'${gCNR_{{H_2}/{B}}}$')
    ax.axhline(reference[1],color='red',linestyle = '--')

    ax.plot(lmbda, gcnrh12, color='orange',label = r'${gCNR_{{H_1}/{H_2}}}$')
    ax.axhline(reference[2],color='orange',linestyle = '--')

    ax.plot(lmbda, gcnrh2a, color='purple',label = r'${gCNR_{{H_2}/{A}}}$')
    ax.axhline(reference[3],color='purple',linestyle = '--')

    ax.set_ylabel(r'${gCNR}$')
    ax.set_xlabel(r'$𝜆$')
    ax.set_xscale('log')
    ax.set_ylim(0,1)

    ax.legend()
    plt.tight_layout()
    plt.show()

    return lmbda[np.argmax(gcnrh2a)]

def gCNRPlot(r1, r2, min, max,ax,median_flag = False,y_flag = False):

    region_r1 = np.ravel(r1)
    region_r2 = np.ravel(r2)

    if median_flag == True:
        log_r1 = processing.imag2uint(region_r1, min, max)
        log_r2 = processing.imag2uint(region_r2, min, max)
    else:
        log_r1 = processing.imag2uint(10 * np.log10(region_r1), min, max)
        log_r2 = processing.imag2uint(10 * np.log10(region_r2), min, max)

    weights = np.ones_like(log_r1) / float(len(log_r1))

    ax.hist(log_r1, bins=bins, range=(0, 255), weights=weights, histtype='step', label=r'${H_1}$')

    ax.hist(log_r2, bins=bins, range=(0, 255), weights=weights, histtype='step', label=r'${H_2}$')

    ax.legend()
    ax.set_ylim(0,0.5)

    if y_flag == True:
        ax.set_ylabel('pixel percentage',fontsize=20)
        y_vals = ax.get_yticks()
        ax.set_yticklabels(['{:d}%'.format(int(x*100)) for x in y_vals])
        pass
    else:
        ax.set_yticks([])
        ax.set_ylabel('')

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
    lmbda = np.logspace(-4,0,50)
    value = []
    for i in range(len(lmbda)):

        value.append(lmbda_search(s,lmbda[i],0.05))

    best = value_plot(lmbda,value)

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

    fig = plt.figure(figsize=(16, 9),constrained_layout=True)
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

    ax = fig.add_subplot(gs[0,0])
    ax.set_axis_off()
    ax.set_title('(a) reference')
    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0], vmax=vmax,
              vmin=rvmin, interpolation='none')
    anote(ax,s_intensity)
    ax = fig.add_subplot(gs[1, 0])
    gCNRPlot(ho_s_1, ho_s_2, rvmin, vmax,ax,y_flag=True)

    ax = fig.add_subplot(gs[0,1])
    textstr = r'(b) $𝜆$ = %.2f,$W$ = %.1f' % (best,speckle_weight)

    ax.set_title(textstr)
    ax.set_axis_off()
    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')
    anote(ax,x_intensity)
    ax = fig.add_subplot(gs[1, 1])
    gCNRPlot(ho_x_1, ho_x_2, rvmin, vmax,ax)

    b_log = median_filter(x_log, size=(3, 3))
    ax = fig.add_subplot(gs[0, 2])

    textstr = '\n'.join((
        r'(c) $𝜆$ = %.2f ' % (best),
        r'$W$ = %.1f,3x3 median' % (speckle_weight)))

    ax.set_title(textstr)
    ax.imshow(b_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')
    anote(ax,x_intensity,median_flag = True)

    ho_b_1 = quality.ROI(*roi['homogeneous'][0], b_log)
    ho_b_2 = quality.ROI(*roi['homogeneous'][1], b_log)

    ar_b = quality.ROI(*roi['background'][0], b_log)

    ax = fig.add_subplot(gs[1, 2])
    gCNRPlot(ho_b_1, ho_b_2, rvmin, vmax,ax, median_flag = True)

    ax = fig.add_subplot(gs[:,3])
    ax.set_title(r'(d) generalized $CNR$ $vs.$ $𝜆$')
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

    ax.semilogx(lmbda, gcnrh1a, color='green', label=r'${gCNR_{{H_1}/{A}}}$')
    ax.axhline(reference[0], color='green', linestyle='--')

    ax.semilogx(lmbda, gcnrh2b, color='red', label=r'${gCNR_{{H_2}/{B}}}$')
    ax.axhline(reference[1], color='red', linestyle='--')

    ax.semilogx(lmbda, gcnrh12, color='orange', label=r'${gCNR_{{H_1}/{H_2}}}$')
    ax.axhline(reference[2], color='orange', linestyle='--')

    ax.semilogx(lmbda, gcnrh2a, color='purple', label=r'${gCNR_{{H_2}/{A}}}$')
    ax.axhline(reference[3], color='purple', linestyle='--')

    ax.set_ylabel(r'${gCNR}$',fontsize=20)
    ax.set_xlabel(r'$𝜆$')

    ax.set_ylim(0.25, 1)
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.legend(loc = 'best',fontsize = 13)
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
