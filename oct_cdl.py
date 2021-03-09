# -*- coding: utf-8 -*-
# @Time    : 2021-03-08 6:10 p.m.
# @Author  : young wang
# @FileName: git_cdl.py
# @Software: PyCharm

from sporco import cnvrep
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

def intensity_norm(data):
    pixels = 255 * (data - data.min()) / (data.max() - data.min())
    return pixels

np.seterr(divide='ignore', invalid='ignore')

# Customize matplotlib
matplotlib.rcParams.update(
    {
        'font.size': 20,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

# define signal & dictionary path
S_PATH = '/Users/youngwang/Desktop/Data/paper/Data/ear'

# load signal & dictionary
with open(S_PATH, 'rb') as f:
    s0 = pickle.load(f).T
    f.close()

K = s0.shape[1] # number of A-line signal
N = s0.shape[0] # length of A-line signgal
M = 1 # state of dictionary

# pre-processing data
for i in range(s0.shape[1]):
    # (1) remove the DC term of each A-line by
    # subtracting the mean of the A-line
    s0[:, i] -= np.mean(s0[:, i])

# (2) remove background noise: minus the frame mean
s0 -= np.mean(s0, axis=1)[:, np.newaxis]
s = copy.deepcopy(s0)

# (3) l2 norm data and save the scaling factor
l2f = prox.norm_l2(s,axis=0).squeeze()
for i in range(s.shape[1]):
    s[:,i] /= l2f[i]


s0_log = 20 * np.log10(abs(s0))
s0_log = intensity_norm(s0_log)

# randomly select one A-line as the dictionary
# dic_index = np.random.choice(s.shape[1],1)
dic_index = 6500 # fixed here for repeatability and reproducibility
# l2 normalize the dictionary
D = s[:,dic_index]

lmbda = 1.34e-1
#
#convert to sporco standard layabout
D = np.reshape(D,(-1,1,M))
#
# uniform random sample the training set from input test, 10%
train_index = np.random.choice(s.shape[1],int(0.1*K),replace=False)
s_train = s[:,train_index]
#
Maxiter = 1000
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                            'MaxMainIter': Maxiter,'RelStopTol': 5e-5, 'AuxVarObj': True,
                                            'RelaxParam': 1.515,'AutoRho': {'Enabled': True}})
# convert to sporco standard layabout
s_train = np.reshape(s_train, (-1,1,len(train_index)))

cri = cnvrep.CDU_ConvRepIndexing(D.shape, s_train)

optx = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1,
            'rho': 8.13e+01,'AuxVarObj': False})

optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
            'rho': 10, 'ZeroMean': False},
            method='cns')
#
#Dictionary support projection and normalisation (cropped).
#Normalise dictionary according to dictionary Y update options.

Dn = cnvrep.Pcn(D, D.shape, cri.Nv, dimN=1, dimC=0, crp=False)

# Update D update options to include initial values for Y and U.
optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(Dn, cri.Cd, cri.M), cri.Nv),
             'U0': np.zeros(cri.shpD + (cri.K,))})
#
#Create X update object.
xstep = cbpdn.ConvBPDN(Dn, s_train, lmbda , optx)
# # the first one is coefficient map
# #Create D update object. with consensus method
dstep = ccmod.ConvCnstrMOD(None, s_train, D.shape, optd, method='cns')
#
t = TicToc()
t.tic()
opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter':Maxiter})
d = dictlrn.DictLearn(xstep, dstep, opt)
D1 = d.solve().squeeze()

D1 = np.roll(D1,-160)
#
D = D.squeeze()

t.toc('DictLearn solve time:')
itsx = xstep.getitstat()
itsd = dstep.getitstat()

fig = plt.figure(figsize=(18, 13),constrained_layout=True)
gs = gridspec.GridSpec(2,3, figure =fig)
ax = fig.add_subplot(gs[0,:])
#
ax.plot(itsx.ObjFun)
ax.set_ylabel('cost function value')
ax.set_xlabel('iteration')
ax.set_title('dictionary learning curve')
#
ax = fig.add_subplot(gs[1,0])
vmax = 255
vmin = 120

ax.set_title('%d dB-%d dB' %(vmax,vmin))
ax.imshow(s0_log, cmap='gray', vmax=vmax, vmin=vmin)
ax.set_aspect(s0_log.shape[1] / s0_log.shape[0])
ax.axvline(x=dic_index,linewidth=2, color='r')
ax.set_axis_off()
#
ax = fig.add_subplot(gs[1,1])
ax.set_title('selected A-line')
ax.plot(abs(D))
ax.set_ylabel('magnitude(a.u.)')
ax.set_xlabel('axial depth(pixels)')


ax = fig.add_subplot(gs[1,2])
ax.set_title('estimated PSF')
ax.plot(abs(D1))
ax.set_ylabel('magnitude(a.u.)')
ax.set_xlabel('axial depth(pixels)')

plt.show()

D1 = D1.reshape(-1,1)

# # with open('/Users/youngwang/Desktop/Data/paper/PSF/learned','wb') as f:
# #     pickle.dump(D1,f)
# #     f.close()
