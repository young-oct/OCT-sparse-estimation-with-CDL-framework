# -*- coding: utf-8 -*-
# @Time    : 2021-03-08 8:27 p.m.
# @Author  : young wang
# @FileName: git_cbpdn.py
# @Software: PyCharm

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sporco import prox
import copy
from sporco import mpiutil
import matplotlib.gridspec as gridspec
from sporco.admm import cbpdn
from pytictoc import TicToc
from skimage.exposure import match_histograms
from PIL import Image

def intensity_norm(data):
    pixels = 255 * (data - data.min()) / (data.max() - data.min())
    return pixels

np.seterr(divide='ignore', invalid='ignore')
# Customize matplotlib
matplotlib.rcParams.update(
    {
        'font.size': 22,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

def evalerr(prm):
    lmbda = prm[0]
    b = cbpdn.ConvBPDN(D0, test, lmbda, opt=opt_par, dimK=None, dimN=1)
    x = b.solve().squeeze()
    x = np.roll(x, np.argmax(D0), axis=0)

    return prox.norm_l1(abs(x[125::]))

# define signal & dictionary path
S_PATH = '/Users/youngwang/Desktop/Data/paper/Data/ear'
D_PATH = '/Users/youngwang/Desktop/Data/paper/PSF/optimal1.34'

# load signal & dictionary
with open(S_PATH, 'rb') as f:
    s0 = pickle.load(f).T
    f.close()

with open(D_PATH, 'rb') as f:
    D0 = pickle.load(f)
    f.close()

# format data as [depth x line], ex[330x10240]

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

# log modulation to construct image
s0_log = 20 * np.log10(abs(s0))
s0_log = intensity_norm(s0_log)

# define a test line
index = 8800
test = s[:, index]
Maxiter = 200
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': False, 'Verbose': False, 'StatusHeader': False,
                                  'MaxMainIter': Maxiter, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                  'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})
# parameter search
t = TicToc()
t.tic()
lrng = np.logspace(-4, 2, 400)
sprm, sfvl, fvmx, sidx = mpiutil.grid_search(evalerr, (lrng,))
lmbda = sprm[0]
print('Minimum ℓ1 error: %5.2f at 𝜆 = %.2e' % (sfvl, lmbda))
t.toc('Lambda search elapsed time is ', restart=True)

fig = plt.figure(figsize=(18, 13),constrained_layout=True)
vmax = 255
vmin = 140

gs = gridspec.GridSpec(3, 2,figure =fig)
ax = fig.add_subplot(gs[0, 0])
ax.set_title('original image: %d-%d' %(vmin,vmax))
ax.imshow(s0_log, cmap='gray', vmax=vmax, vmin=vmin)
ax.set_aspect(s0_log.shape[1] / s0_log.shape[0])
ax.axvline(x=index,linewidth=2, color='r')
ax.set_axis_off()

ax = fig.add_subplot(gs[0, 1])
ax.semilogx(lrng, fvmx)
ax.set_ylabel('reconstruction error')
ax.set_xlabel('𝜆')
ax.set_title('initial 𝜆 = %.5f' % float(sprm[0]))

b = cbpdn.ConvBPDN(D0, test, lmbda, opt=opt_par, dimK=None, dimN=1)
x = b.solve()
its = b.getitstat()
x = x.squeeze()
x = np.roll(x, np.argmax(D0), axis=0)

sparsity = 100 * np.count_nonzero(x) / x.size

ax = fig.add_subplot(gs[1, 0])
ax.plot(abs(test))

ax.set_xlabel('axial depth (pixels)')
ax.set_ylabel('magnitude(a.u.)')
ax.set_xlabel('axial depth(pixels)')
ax.set_title('original A-line')

ax = fig.add_subplot(gs[1, 1])
ax.plot(abs(x))
ax.set_xlabel('axial depth (pixels)')
ax.set_ylabel('magnitude(a.u.)')
ax.set_xlabel('axial depth(pixels)')
ax.set_title('spare A-line A-line(sparsity =%3.f %%)' % sparsity)

ax = fig.add_subplot(gs[2, :])

ax.plot(its.ObjFun)
ax.set_ylabel('cost function value')
ax.set_xlabel('iteration')
ax.set_title('sparse coding curve')

plt.show()

index = 1250
Maxiter = 20
opt_par['FastSolve'] = False
s_line = s[:,index]
b = cbpdn.ConvBPDN(D0, s_line, lmbda, opt=opt_par, dimK=None, dimN=1)
x = b.solve()
its = b.getitstat()
x = x.squeeze()
x = np.roll(x, np.argmax(D0), axis=0)
x_line = abs(x)
scale = np.max(abs(x))/np.max(abs(s_line))

Maxiter = 20
b = cbpdn.ConvBPDN(D0, s, lmbda, opt=opt_par, dimK=1, dimN=1)
x = b.solve()
its = b.getitstat()
x = x.squeeze()
x = np.roll(x, np.argmax(D0), axis=0)

for j in range(x.shape[1]):
    x[:, j] *= l2f[j]

sparisty = np.count_nonzero(x) * 100 / x.size
x_log = x.T

sparse = np.zeros((330, 10240))
# rescale the sparse solution
for j in range(s.shape[1]):
    x_log[j, :] = abs(x_log[j, :]) / scale

x_line = x_log[index, :]
x_log = 20 * np.log10(abs(x_log))
eps = 1e-14

x_log = np.where(x_log < 20 * np.log10(eps), 20 * np.log10(eps), x_log)
x_log = intensity_norm(x_log).T

temp = match_histograms(x_log,s0_log,multichannel=False)
sparse = np.where(temp <= np.min(temp), 0,temp)

vmin = 140
fig = plt.figure(figsize=(18, 13),constrained_layout=True)
gs = gridspec.GridSpec(2, 2,figure =fig)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(s0_log, cmap='gray', vmax=vmax, vmin=vmin)
ax.set_title('original image: %d -%d'% (vmin, vmax))
ax.set_aspect(s0_log.shape[1] / s0_log.shape[0])
ax.axvline(x=index,linewidth=1, color='r',linestyle = '--')
ax.set_axis_off()

vmin = 0
ax = fig.add_subplot(gs[1, 0])
ax.set_title('sparse image:%d - %d'% (vmin, vmax))
ax.imshow(sparse, cmap='gray', vmax=vmax, vmin=0)
ax.set_aspect(sparse.shape[1] / sparse.shape[0])
ax.axvline(x=index,linewidth=1, color='r',linestyle = '--')
ax.set_axis_off()

ax = fig.add_subplot(gs[0, 1])
ax.plot(abs(s0[:,index]),label='original A-line',linestyle = '--')
ax.plot(abs(x_line),label='sparse A-line')
ax.set_xlabel('axial depth(pixel)')
ax.set_ylabel('magnitude(a.u.)')
ax.set_title('A-line')
ax.legend(loc = 'best')
axins = ax.inset_axes([0.35, 0.2, 0.6, 0.55])
axins.set_xticks([])
axins.set_yticks([])
axins.plot(abs(s0[:,index]),linestyle = '--')
axins.plot(abs(x_line))
axins.set_xlim(35, 65)
axins.set_ylim(0, 73000)

textstr_an = 'anterior''\n''sidelobe'
textstr_po = 'posterior''\n''sidelobe'

axins.annotate(textstr_an, xy=(45, 30000),  xycoords='data',
            xytext=(43, 60000), textcoords='data',fontsize=20,
            color='red',fontname ='Arial',
            arrowprops=dict(facecolor='red', shrink=0.025),
            horizontalalignment='right', verticalalignment='top')

axins.annotate(textstr_po, xy=(57, 47000),  xycoords='data',
            xytext=(58, 71000), textcoords='data', fontsize=20,
            color='red',fontname ='Arial',
            arrowprops=dict(facecolor='red', shrink=0.025),
            horizontalalignment='left', verticalalignment='top')

ax.indicate_inset_zoom(axins)

ax = fig.add_subplot(gs[1, 1])
ax.hist(np.ravel(s0_log),bins = 128, density = True,log=True,histtype = 'step',label = 'original image')
ax.hist(np.ravel(sparse), bins = 128,density = True,log=True,histtype = 'step',label = 'sparse image')
ax.legend(loc = 'best')
ax.set_title('histogram')
ax.set_xlabel('intensity')
ax.yaxis.set_visible(False)

plt.show()


# # save as bmp
# image = Image.fromarray(s0_log.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/original.bmp')
#
# image = Image.fromarray(sparse.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/sparse.bmp')
