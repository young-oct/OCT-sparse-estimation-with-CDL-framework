# -*- coding: utf-8 -*-
# @Time    : 2021-05-28 12:02 p.m.
# @Author  : young wang
# @FileName: Guassin_compare.py
# @Software: PyCharm

from sporco import cnvrep
from pathlib import Path

from sporco.dictlrn import dictlrn
from sporco import prox
from sporco.admm import cbpdn,ccmod
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import copy
import matplotlib.gridspec as gridspec
from pytictoc import TicToc
from misc.processing import getAline
from misc import processing
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from misc import processing, quality, annotation
import matplotlib.gridspec as gridspec
from tabulate import tabulate


def get_PSF(s):

    lmbda = 0.1
    l2f, snorm = processing.to_l2_normed(s)

    K = snorm.shape[1]  # number of A-line signal
    M = 1  # state of dictionary

    # randomly select one A-line as the dictionary
    # dic_index = np.random.choice(s.shape[1],1)
    dic_index = int(4500 / decimation_factor)  # fixed here for repeatability and reproducibility
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

    shift = 165
    D1 = np.roll(D1, -shift)

    D1 = D1.reshape(-1, 1)

    return D1


def load_data(S_PATH, decimation_factor):
    # check if such file exists
    if Path(S_PATH).is_file():
        # load data & dictionary
        with open(S_PATH, 'rb') as f:
            s = pickle.load(f).T
            f.close()
        # (2) remove background noise: minus the frame mean
        s = s - np.mean(s, axis=1)[:, np.newaxis]
        # (3) sample every decimation_factor line,
        s = s[:, ::decimation_factor]
        return s
    else:
        raise Exception("Dataset %s not found")

from misc.processing import getAline
# Define ROIs
roi = {}
width, height = (20, 10)
roi['artifact'] = [[212, 142, int(width * 1.2), int(height * 1.2)]]
roi['background'] = [[390, 247, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[212, 165, int(width * 1.2), int(height * 1.2)],
                      [390, 225, int(width * 1.2), int(height * 1.2)]]

# Module level constants
eps = 1e-14
if __name__=='__main__':

    file = r'/Users/youngwang/Desktop/Data/2020.12.11/Finger/2020-Dec-11  04.47.28 PM.bin'
    std,start = 365, 420
    index = 256
    # window_flag = 'wiener'
    window_flag = 'square'

    getAline(file,std,start,window_flag=window_flag)

    ref_path = '/Users/youngwang/Desktop/Master/seminar/test_data/ref'
    gaussian_path = '/Users/youngwang/Desktop/Master/seminar/test_data/window'

    decimation_factor = 20
    s_ref = load_data(ref_path,decimation_factor)
    _, temp = processing.to_l2_normed(s_ref)
    s_line = abs(temp[:,index])

    s_gaussian = load_data(gaussian_path, decimation_factor)
    _, temp = processing.to_l2_normed(s_gaussian)
    w_line = abs(temp[:, index])

    D = get_PSF(s_ref)

    # Image processing and display paramaters
    speckle_weight = 0.01

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
    # Load the example dataset

    lmbda = 0.03
    w_lmbda = 0.02

    x,x_line = processing.make_sparse_representation(s_ref,D, lmbda,w_lmbda,
                                              speckle_weight, Line=True,index =index)

    rvmin = 5  # dB
    vmax = 55  # dB
    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s_ref))

    sg_log = 20 * np.log10(abs(s_gaussian))

    x_log = 20 * np.log10(abs(x))

    s_intensity = abs(s_ref) ** 2
    sg_intensity = abs(s_gaussian) ** 2

    x_intensity = abs(x) ** 2

    ho_s_1 = quality.ROI(*roi['homogeneous'][0], s_intensity)
    ho_s_2 = quality.ROI(*roi['homogeneous'][1], s_intensity)

    ho_sg_1 = quality.ROI(*roi['homogeneous'][0], sg_intensity)
    ho_sg_2 = quality.ROI(*roi['homogeneous'][1], sg_intensity)

    ho_x_1 = quality.ROI(*roi['homogeneous'][0], x_intensity)
    ho_x_2 = quality.ROI(*roi['homogeneous'][1], x_intensity)

    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_sg = quality.ROI(*roi['artifact'][0], sg_intensity)

    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_sg = quality.ROI(*roi['background'][0], sg_intensity)

    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    # fig = plt.figure(figsize=(18, 13), constrained_layout = True)
    fig = plt.figure(figsize=(18, 13))

    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title('(a) reference', fontsize= 28)

    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] ), textcoords='data', fontsize=30,
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
    ax.annotate(text, xy=(roi['background'][0][0]+width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2*width, roi['background'][0][1] + 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_axis_off()

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
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ar_s)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(ho_s_2, ar_s)),
        r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(ho_s_2, ba_s)),
        r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(ho_s_1, ho_s_2))))
    ax.text(0.625, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1,0])
    ax.plot(s_line)
    ax.set_ylim(0,1.1*np.max(s_line))
    ax.set_xlabel('axial depth [pixels]', fontname='Arial')

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('(b) %s' %window_flag,fontsize=28)

    ax.imshow(sg_log, 'gray', aspect=sg_log.shape[1] / sg_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

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
        r'${SNR_{{H_2}/B}}$: %.1f $dB$' % (quality.SNR(ho_sg_2, ba_sg)),
        r'${C_{{H_2}/B}}$: %.1f $dB$' % (quality.Contrast(ho_sg_2, ba_sg)),
        r'${C_{{H_1}/{H_2}}}$: %.1f $dB$' % (quality.Contrast(ho_sg_1, ho_sg_2))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_sg_1, ar_sg)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(ho_sg_2, ar_sg)),
        r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(ho_sg_2, ba_sg)),
        r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(ho_sg_1, ho_sg_2))))
    ax.text(0.625, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')

    ax = fig.add_subplot(gs[1,1])
    ax.plot(w_line)
    ax.set_yticks([])
    ax.set_ylim(0,1.1*np.max(s_line))
    ax.set_xlabel('axial depth [pixels]', fontname='Arial')

    ax = fig.add_subplot(gs[0,2])

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin,interpolation='none')

    text = r'${H_{1}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 50, roi['homogeneous'][0][1] + 50), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][1][0], roi['homogeneous'][1][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][1][0] - 50, roi['homogeneous'][1][1] ), textcoords='data', fontsize=30,
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
    ax.annotate(text, xy=(roi['background'][0][0]+width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2*width, roi['background'][0][1] + 40), textcoords='data', fontsize=30,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    ax.set_title('(c) 𝜆 = %.2f \n $\omega$ = %.2f' % (lmbda, speckle_weight),fontsize = 28)

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
        r'${SNR_{{H_2}/B}}$: %.1f $dB$' % (quality.SNR(ho_x_2, ba_x)),
        r'${C_{{H_2}/B}}$: %.1f $dB$' % (quality.Contrast(ho_x_2, ba_x)),
        r'${C_{{H_1}/{H_2}}}$: %.1f $dB$' % (quality.Contrast(ho_x_1, ho_x_2))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')

    textstr = '\n'.join((
        r'${gCNR_{{H_1}/{A}}}$: %.2f' % (quality.log_gCNR(ho_x_1, ar_x)),
        r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(ho_x_2, ar_x)),
        r'${gCNR_{{H_2}/B}}$: %.2f' % (quality.log_gCNR(ho_x_2, ba_x)),
        r'${gCNR_{{H_1}/{H_2}}}$: %.2f' % (quality.log_gCNR(ho_x_1, ho_x_2))))
    ax.text(0.625, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', fontname='Arial', color='white')
    ax = fig.add_subplot(gs[1,2])
    ax.plot(x_line)
    ax.set_yticks([])
    ax.set_ylim(0,1.1*np.max(s_line))
    ax.set_xlabel('axial depth [pixels]', fontname='Arial')

    plt.tight_layout()
    plt.show()
