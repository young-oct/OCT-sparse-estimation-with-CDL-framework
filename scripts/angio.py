# -*- coding: utf-8 -*-
# @Time    : 2022-03-18 9:53 p.m.
# @Author  : young wang
# @FileName: angio.py
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
import time


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
    train_index = np.random.choice(snorm.shape[1], int(0.2 * K), replace=False)
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


# Module level constants
eps = 1e-14

if __name__ == '__main__':
    t = time.process_time()
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
    d_lmbda = 0.1


    with open('../data/middle_ear-aline.bin', 'rb') as f:
        alineData = np.fromfile(f, dtype=np.complex64).reshape([4, 2560, 330])

    raw = alineData[0, :, :].T

    s = processing.mean_remove(raw, decimation_factor=1)
    D = get_PSF(s, d_lmbda)
    lmbda = 0.028
    w_lmbda = 0.05
    #
    x = processing.make_sparse_representation(s, D, lmbda, w_lmbda, speckle_weight, rvmin=55, vmax=100)
    #
    # Generate log intensity arrays
    s_log = 20 * np.log10(abs(s))
    x_log = 20 * np.log10(abs(x))

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 9))

    rvmin, vmax = 55, 100  # dB

    ax[0].set_title('original')
    ax[0].imshow(np.flipud(s_log), 'gray', aspect=s_log.shape[1] / s_log.shape[0],
                 vmax=vmax, vmin=rvmin, interpolation='none')

    ax[1].set_title('sparse')
    ax[1].imshow(np.flipud(x_log), 'gray', aspect=x_log.shape[1] / x_log.shape[0],
                 vmax=vmax, vmin=rvmin, interpolation='none')

    plt.tight_layout()
    plt.show()



