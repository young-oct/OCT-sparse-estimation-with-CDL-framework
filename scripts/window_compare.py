# -*- coding: utf-8 -*-
# @Time    : 2021-06-09 10:49 p.m.
# @Author  : young wang
# @FileName: window_compare.py
# @Software: PyCharm

from misc import processing, quality, annotation
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.dictlrn import dictlrn
from sporco.admm import cbpdn, ccmod
from sporco import cnvrep
import pickle
from scipy.ndimage import median_filter
from sporco.admm import cbpdn

def get_PSF(s, lmbda):
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

    ax.hist(log_r1, bins=bins, range=(0, 255), weights=weights, histtype='step', label=r'${H_2}$')

    ax.hist(log_r2, bins=bins, range=(0, 255), weights=weights, histtype='step', label=r'${A}$')

    ax.legend()
    ax.set_ylim(0,1.05)

    if y_flag == True:
        ax.set_ylabel('pixel percentage',fontsize=20)
        y_vals = ax.get_yticks()
        ax.set_yticklabels(['{:d}%'.format(int(x*100)) for x in y_vals])
        pass
    else:
        ax.set_yticks([])
        ax.set_ylabel('')

    return ax

def anote(ax,s,median_flag =False):

    text = r'${A}$'
    ax.annotate(text, xy=(roi['artifact'][0][0], roi['artifact'][0][1]), xycoords='data',
                xytext=(roi['artifact'][0][0] - 100, roi['artifact'][0][1] - 45), textcoords='data', fontsize=legend_font,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='left', verticalalignment='top')

    text = r'${H_{2}}$'
    ax.annotate(text, xy=(roi['homogeneous'][0][0], roi['homogeneous'][0][1] + height), xycoords='data',
                xytext=(roi['homogeneous'][0][0] - 55, roi['homogeneous'][0][1]+height+8), textcoords='data', fontsize=legend_font,
                color='white', fontname='Arial',
                arrowprops=dict(facecolor='white', shrink=0.025),
                horizontalalignment='right', verticalalignment='top')

    text = r'${B}$'
    ax.annotate(text, xy=(roi['background'][0][0] + width, roi['background'][0][1] + height), xycoords='data',
                xytext=(roi['background'][0][0] + 2 * width, roi['background'][0][1] + 40), textcoords='data',
                fontsize=legend_font,
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


    h2 = quality.ROI(*roi['homogeneous'][0], s)
    ba = quality.ROI(*roi['background'][0], s)
    ar = quality.ROI(*roi['artifact'][0], s)

    if median_flag == True:

        textstr =r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(h2, ar,improvement=True))
    else:
        textstr = '\n'.join((
            r'${SNR_{{H_2}/B}}$: %.1f $dB$' % (quality.SNR(h2, ba)),
            r'${gCNR_{{H_2}/{A}}}$: %.2f' % (quality.log_gCNR(h2, ar,improvement=False))))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=legend_font,
            verticalalignment='top', fontname='Arial', color='white')
    return ax

def zoomshow(ax,image):
    zoom_factor = 10
    axins = ax.inset_axes([300, 40, width*zoom_factor, height*zoom_factor], transform=ax.transData)
    axins.imshow(image, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.spines['left'].set_color('green')
    axins.spines['right'].set_color('green')
    axins.spines['top'].set_color('green')
    axins.spines['bottom'].set_color('green')

    x1,x2 =roi['artifact'] [0][0], int(roi['artifact'] [0][0]+width)
    y1,y2 =roi['artifact'] [0][1], int(roi['artifact'] [0][1]+height)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor='green')

    axins = ax.inset_axes([10, 235, width*zoom_factor, height*zoom_factor], transform=ax.transData)
    axins.imshow(image, cmap='gray', vmax=vmax, vmin=rvmin, interpolation='none')
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    axins.spines['left'].set_color('red')
    axins.spines['right'].set_color('red')
    axins.spines['top'].set_color('red')
    axins.spines['bottom'].set_color('red')

    x1, x2 = roi['homogeneous'][0][0], int(roi['homogeneous'][0][0] + width)
    y1, y2 = roi['homogeneous'][0][1], int(roi['homogeneous'][0][1] + height)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor='red')
    return ax

# Define ROIs
roi = {}
width, height = (20, 10)
roi['artifact'] = [[212, 142, int(width * 1.2), int(height * 1.2)]]
roi['background'] = [[390, 260, int(width * 1.2), int(height * 1.2)]]
roi['homogeneous'] = [[390, 230, int(width * 1.2), int(height * 1.2)]]

# Module level constants
eps = 1e-14
legend_font = 20
bins = 32
if __name__ == '__main__':
    # Image processing and display paramaters
    speckle_weight = 0.1
    rvmin, vmax = 5, 55  # dB

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

    start, decimation_factor = 420, 20
    # gaussian std
    std = 292
    d_lmbda = 0.1

    raw = processing.load_raw('/Users/youngwang/Desktop/github/data/finger(raw).npz')

    s_r = processing.mean_remove(processing.Aline_R(raw, start), decimation_factor)
    s_g = processing.mean_remove(processing.Aline_G(raw, start, std), decimation_factor)
    s = processing.mean_remove(processing.Aline_H(raw, start), decimation_factor)

    # D = get_PSF(s,d_lmbda)

    with open('../data/PSF/finger', 'rb') as f:
        D = pickle.load(f)
        f.close()

    lmbda = 0.028
    w_lmbda = 0.05

    x = processing.make_sparse_representation(s, D, lmbda, w_lmbda, speckle_weight)

    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))

    sr_log = 20 * np.log10(abs(s_r))
    sg_log = 20 * np.log10(abs(s_g))

    sr_intensity = abs(s_r) ** 2
    sg_intensity = abs(s_g) ** 2
    s_intensity = abs(s) ** 2
    x_intensity = abs(x) ** 2

    ho_r_2 = quality.ROI(*roi['homogeneous'][0], sr_intensity)

    ho_g_2 = quality.ROI(*roi['homogeneous'][0], sg_intensity)

    ho_s_2 = quality.ROI(*roi['homogeneous'][0], s_intensity)

    ho_x_2 = quality.ROI(*roi['homogeneous'][0], x_intensity)

    ar_r = quality.ROI(*roi['artifact'][0], sr_intensity)
    ar_g = quality.ROI(*roi['artifact'][0], sg_intensity)
    ar_s = quality.ROI(*roi['artifact'][0], s_intensity)
    ar_x = quality.ROI(*roi['artifact'][0], x_intensity)

    ba_r = quality.ROI(*roi['background'][0], sr_intensity)
    ba_g = quality.ROI(*roi['background'][0], sg_intensity)
    ba_s = quality.ROI(*roi['background'][0], s_intensity)
    ba_x = quality.ROI(*roi['background'][0], x_intensity)

    fig = plt.figure(figsize=(16, 9),constrained_layout=True)

    gs = fig.add_gridspec(ncols=4, nrows=1)

    ax = fig.add_subplot(gs[0])
    ax.set_title('(a) no window')

    ax.imshow(sr_log, 'gray', aspect=sr_log.shape[1] / sr_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')
    zoomshow(ax,sr_log)

    anote(ax,sr_intensity)

    ax = fig.add_subplot(gs[1])
    ax.set_title('(b) Gaussian window')

    ax.imshow(sg_log, 'gray', aspect=sg_log.shape[1] / sg_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    zoomshow(ax, sg_log)
    anote(ax,sg_intensity)

    ax = fig.add_subplot(gs[2])
    ax.set_title('(c) Hann window')

    ax.imshow(s_log, 'gray', aspect=s_log.shape[1] / s_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    zoomshow(ax, s_log)
    anote(ax,s_intensity)
    
    ax = fig.add_subplot(gs[3])
    textstr = r'(d) $𝜆$ = %.2f,$W$ = %.1f' % (lmbda,speckle_weight)
    ax.set_title(textstr)

    ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
              vmax=vmax, vmin=rvmin, interpolation='none')

    zoomshow(ax, x_log)
    anote(ax,x_intensity)
    plt.show()

    # fig,ax = plt.subplots(figsize=(16,9))
    # ax.set_title('𝜆 = %.2f, $W$ = %.1f'
    #              % (lmbda, speckle_weight), fontsize=25)
    #
    # ax.imshow(x_log, 'gray', aspect=x_log.shape[1] / x_log.shape[0],
    #           vmax=vmax, vmin=rvmin, interpolation='none')
    # ax.set_axis_off()
    # plt.show()