# -*- coding: utf-8 -*-
# @Time    : 2021-06-09 10:49 p.m.
# @Author  : young wang
# @FileName: window_compare.py
# @Software: PyCharm


import numpy as np
import uuid
from math import ceil
import sys
from matplotlib import pyplot as plt
from numpy.fft import fft, fftshift, ifft
import pickle
from pytictoc import TicToc
from scipy import signal
from scipy.signal.signaltools import wiener
import copy
from misc import processing
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.dictlrn import dictlrn
from sporco.admm import cbpdn,ccmod
from sporco import cnvrep
from pytictoc import TicToc
import matplotlib.gridspec as gridspec
import pickle
from sporco.admm import cbpdn
from misc import processing


def get_PSF(s,lmbda):
    l2f, snorm = processing.to_l2_normed(s)

    K = snorm.shape[1]  # number of A-line signal
    M = 1  # state of dictionary

    # randomly select one A-line as the dictionary
    # dic_index = np.random.choice(s.shape[1],1)
    dic_index = int(s.shape[1] / 2)  # fixed here for repeatability and reproducibility
    # l2 normalize the dictionary
    D = snorm[:, dic_index]

    # convert to sporco standard layabout
    D = np.reshape(D, (-1, 1, M))

    # uniform random sample the training set from input test, 10%
    train_index = np.random.choice(snorm.shape[1], int(0.25 * K), replace=False)
    s_train = snorm[:, train_index]
    #
    Maxiter = 1000

    # convert to sporco standard layabout
    s_train = np.reshape(s_train, (-1, 1, len(train_index)))

    cri = cnvrep.CDU_ConvRepIndexing(D.shape, s_train)

    optx = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1,
                                   'rho': 8.13e+01, 'AuxVarObj': False})

    optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
                                      'rho': 10, 'ZeroMean': False},
                                     method='cns')
    #
    # Dictionary support projection and normalisation (cropped).
    # Normalise dictionary according to dictionary Y update options.

    Dn = cnvrep.Pcn(D, D.shape, cri.Nv, dimN=1, dimC=0, crp=False)

    # Update D update options to include initial values for Y and U.
    optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(Dn, cri.Cd, cri.M), cri.Nv),
                 'U0': np.zeros(cri.shpD + (cri.K,))})
    #
    # Create X update object.
    xstep = cbpdn.ConvBPDN(Dn, s_train, lmbda, optx)
    # # the first one is coefficient map
    # #Create D update object. with consensus method
    dstep = ccmod.ConvCnstrMOD(None, s_train, D.shape, optd, method='cns')
    #
    opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter': Maxiter})
    d = dictlrn.DictLearn(xstep, dstep, opt)

    D1 = d.solve().squeeze()
    shift = np.argmax(abs(D1)) - 165
    D1 = np.roll(D1, -shift)

    D1 = D1.reshape(-1, 1)
    return D1

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

    start,decimation_factor = 420,20
    #gaussian std
    std = 50
    d_lmbda = 0.1

    raw = processing.load_raw('/Users/youngwang/Desktop/github/data/finger(raw).npz')

    s_r = processing.mean_remove(processing.Aline_R(raw,start),decimation_factor)
    s_g = processing.mean_remove(processing.Aline_G(raw,start,std),decimation_factor)
    s = processing.mean_remove(processing.Aline_H(raw,start),decimation_factor)

    # D = get_PSF(s,d_lmbda)

    with open('../data/PSF/finger', 'rb') as f:
        D = pickle.load(f)
        f.close()

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
    ax.set_title('(b) gaussian window\n std =%d' % std, fontsize=28)

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



