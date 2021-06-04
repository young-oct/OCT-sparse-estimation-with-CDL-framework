# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 4:21 p.m.
# @Author  : young wang
# @FileName: oct_cdl.py
# @Software: PyCharm


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



# Module level constants
eps = 1e-14
lmbda = 1e-1

D0_PATH = '../Data/PSF/measured'

with open(D0_PATH, 'rb') as f:
    D0 = pickle.load(f)
    f.close()

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 12
        }
    )

    # s, D,path = load_data('finger', decimation_factor=20)
    file_name = ['nail']
    for i in range(len(file_name)):
        decimation_factor = 20
        s = processing.load_data(file_name[i], decimation_factor=decimation_factor, data_only= True)
        l2f, snorm = processing.to_l2_normed(s)

        K = snorm.shape[1]  # number of A-line signal
        N = snorm.shape[0]  # length of A-line signgal
        M = 1  # state of dictionary

        # randomly select one A-line as the dictionary
        # dic_index = np.random.choice(s.shape[1],1)
        dic_index = int(6500/decimation_factor)  # fixed here for repeatability and reproducibility
        # l2 normalize the dictionary
        D = snorm[:, dic_index]

        # convert to sporco standard layabout
        D = np.reshape(D, (-1, 1, M))

        #
        # uniform random sample the training set from input test, 10%
        train_index = np.random.choice(snorm.shape[1], int(0.25 * K), replace=False)
        s_train = snorm[:, train_index]
        #
        Maxiter = 1000
        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                          'MaxMainIter': Maxiter, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})
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
        t = TicToc()
        t.tic()
        opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter': Maxiter})
        d = dictlrn.DictLearn(xstep, dstep, opt)
        D1 = d.solve().squeeze()

        shift = np.argmax(abs(D1)) - np.argmax(abs(D0))
        D1 = np.roll(D1, -shift)
        #
        D = D.squeeze()

        t.toc('DictLearn solve time:')
        itsx = xstep.getitstat()
        itsd = dstep.getitstat()

        fig = plt.figure(figsize=(18, 13), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        ax = fig.add_subplot(gs[0, :])
        #
        ax.plot(itsx.ObjFun)
        ax.set_ylabel('cost function value')
        ax.set_xlabel('iteration')
        ax.set_title('dictionary learning curve')
        #
        ax = fig.add_subplot(gs[1, 0])
        rvmin, vmax = 5, 55  # dB

        s_log = 20 * np.log10(abs(s))
        s_log = processing.imag2uint(s_log, rvmin, vmax)

        vmin, vmax = 0,255
        ax.set_title(file_name[i]+' %d dB-%d dB' % (vmax, vmin))
        ax.imshow(s_log, cmap='gray', vmax=vmax, vmin=vmin)
        ax.set_aspect(s_log.shape[1] / s_log.shape[0])
        ax.axvline(x=dic_index, linewidth=2, color='r')
        ax.set_axis_off()
        #
        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('selected A-line')
        ax.plot(abs(D))
        ax.set_ylabel('magnitude(a.u.)')
        ax.set_xlabel('axial depth(pixels)')

        ax = fig.add_subplot(gs[1, 2])
        ax.set_title('estimated PSF')
        ax.plot(abs(D1))
        ax.set_ylabel('magnitude(a.u.)')
        ax.set_xlabel('axial depth(pixels)')

        plt.show()

        D1 = D1.reshape(-1, 1)

        D_PATH = '../Data/PSF/' + file_name[i]

        # with open(D_PATH,'wb') as f:
        #     pickle.dump(D1,f)
        #     f.close()
        #



