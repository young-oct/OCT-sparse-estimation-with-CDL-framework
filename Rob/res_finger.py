# -*- coding: utf-8 -*-
# @Time    : 2021-04-07 11:57 p.m.
# @Author  : young wang
# @FileName: res_finger.py
# @Software: PyCharm


'''this script shows an indirect evidence that the sparse image is super resolution
version of the original image through 1d convolution'''

import sys

# sys.path.append('/Users/youngwang/Desktop/young-research/Sparse reconstruction/Sporco/thesis/misc')

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sporco import prox
import copy
from sporco.admm import cbpdn
# from misc.postprocessing import intensity_norm
from skimage.exposure import match_histograms

def intensity_norm(data):
    pixels = 255 * (data - data.min()) / (data.max() - data.min())

    return pixels

np.seterr(divide='ignore', invalid='ignore')
# Customize matplotlib
matplotlib.rcParams.update(
    {
        'font.size': 17,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

# define signal & dictionary path
S_PATH = '/Users/youngwang/Desktop/Data/paper/Data/finger'
# D_PATH = '/Users/youngwang/Desktop/Data/paper/PSF/optimal1.05'
D_PATH = '/Users/youngwang/Desktop/Data/paper/PSF/finger_1.05'
PSF_PATH = '/Users/youngwang/Desktop/Data/paper/PSF/psf'


# load signal & dictionary
with open(S_PATH, 'rb') as f:
    original = pickle.load(f)
    f.close()

with open(D_PATH, 'rb') as f:
    D = pickle.load(f)
    f.close()

with open(PSF_PATH, 'rb') as f:
    psf = pickle.load(f)
    f.close()

# format data as [depth x line], ex[330x10240]
original = original.T
s = copy.deepcopy(original)

# pre-processing data
for i in range(s.shape[1]):
    # (1) remove the DC term of each A-line by
    # subtracting the mean of the A-line
    s[:, i] -= np.mean(s[:, i])

# (2) remove background noise: minus the frame mean
s -= np.mean(s, axis=1)[:, np.newaxis]
s_removal = copy.deepcopy(s)

# (3) l2 norm data and save the scaling factor
l2f = prox.norm_l2(s_removal,axis=0).squeeze()
for i in range(s_removal.shape[1]):
    s_removal[:,i] /= l2f[i]

###plot mirror image and selected A-line
original_log = 20 * np.log10(abs(original.T))
s_removal_log = 20 * np.log10(abs(s.T))

# define a test line
index = 4000

s_line = s_removal[:, index]
lmbda = 0.03
svmin = 70
rvmin = 140

Maxiter = 20
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                  'MaxMainIter': Maxiter, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                  'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

b = cbpdn.ConvBPDN(D, s_line, lmbda, opt=opt_par, dimK=None, dimN=1)
x = b.solve()
x = x.squeeze()
x = np.roll(x, np.argmax(D), axis=0)
scale = np.max(abs(x)) / np.max(abs(s_line))

eps = 1e-14

s_log_norm = intensity_norm(s_removal_log).T

s_line = np.abs(s_removal[:,index])
b = cbpdn.ConvBPDN(D, s_removal, lmbda, opt=opt_par, dimK=1, dimN=1)
x = b.solve()
x = x.squeeze()
r_line = b.reconstruct().squeeze()
x = np.roll(x, np.argmax(D), axis=0)

x_line = abs(x[:, index]) / scale

r = np.zeros(x.shape, complex)
# r_line = np.zeros(x.shape, complex)
# r_log = np.zeros(x.shape)
for j in range(x.shape[1]):
    # r_line[:,j] = np.convolve(x[:, j],D.squeeze(),'same')
    x[:, j] *= l2f[j]
    r[:, j] = r_line[:,j]*l2f[j]
    # r[:,j] = np.convolve(x[:, j],D.squeeze(),'same')

# r = np.where(r<=0,0,r)
    # r_log[:,j] = np.convolve(r_line[:,j],D.squeeze(),'same' )

# plt.plot(abs(s[:,index]), label='s')
# # plt.plot(abs(x[:,index]), label='x')
# plt.plot(abs(r[:,index]), label='r')
# plt.show()

# x_l = abs(x[:,5000])
# r_l = abs(r_log[:,5000])

x_log = x.T

# rescale the sparse solution
for j in range(s.shape[1]):
    x_log[j, :] = abs(x_log[j, :]) / scale

x_log = 20 * np.log10(abs(x_log))
r_log = 20 * np.log10(abs(r))
# r_log = np.where(r_log < 20 * np.log10(eps), 20 * np.log10(eps), r_log)

r_log = intensity_norm(r_log)
# # # display rescaling, forcing -inf to be a 20*np.log10(esp)
x_log_correction = np.where(x_log < 20 * np.log10(eps), 20 * np.log10(eps), x_log)

x_log_norm = intensity_norm(x_log_correction).T

# temp_correction = np.where(temp_log < 20 * np.log10(eps), 20 * np.log10(eps), temp_log)

match = match_histograms(x_log_norm,s_log_norm,multichannel=False)
sparse = np.where(match <= np.min(match),0,match)

vmax = 255

aspect = sparse.shape[1]/sparse.shape[0]
#
fig = plt.figure(constrained_layout=True,figsize=(16,9))
gs = fig.add_gridspec(ncols=4, nrows=2)
ax = fig.add_subplot(gs[0,0])

ax.imshow(s_log_norm, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin)
ax.set_axis_off()
ax.set_title('reference: %d-%d' % (vmax, rvmin))

ax.axvline(x=index,ymax = 1, ymin=0.5, linewidth=2, color='orange', linestyle='--')

axins = ax.inset_axes([0.45,0.55,0.52, 0.42])
axins.imshow(s_log_norm,'gray', aspect=aspect, vmax=vmax, vmin=rvmin,origin="lower")
x1, x2, y1, y2 = 2000, 4000, 225, 145
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins, edgecolor="red")

ax = fig.add_subplot(gs[1,0])

ax.plot(s_line)
ax.set_xlabel('axial depth [pixels]',fontname ='Arial')
ax.set_ylabel('normalized intensity [a.u.]',fontname ='Arial')
axins = ax.inset_axes([0.6, 0.3, 0.37, 0.67])
axins.set_xticks([])
axins.set_yticks([])
axins.plot(s_line)

axins.set_xlim(140, 200)
axins.set_ylim(0, 0.25)

ax.indicate_inset_zoom(axins)

aspect = sparse.shape[1]/sparse.shape[0]

ax = fig.add_subplot(gs[0, 1])
ax.imshow(sparse, 'gray', aspect=aspect, vmax=vmax, vmin=svmin)
ax.axvline(x=index,ymax = 1, ymin=0.5, linewidth=2, color='orange', linestyle='--')
ax.set_title('𝜆 = %.3f: %d-%d' % (lmbda, vmax, svmin))
ax.set_axis_off()
axins = ax.inset_axes([0.45,0.55,0.52, 0.42])
axins.imshow(sparse,'gray', aspect=aspect, vmax=vmax, vmin=svmin,origin="lower")
x1, x2, y1, y2 = 2000, 4000, 225, 145
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins, edgecolor="red")

# aspect = width / height
ax = fig.add_subplot(gs[1, 1])

ax.set_yticks([])
ax.plot(x_line)
axins = ax.inset_axes([0.6, 0.3, 0.37, 0.67])
axins.set_xticks([])
axins.set_yticks([])
axins.plot(x_line)

axins.set_xlim(140, 200)
axins.set_ylim(0, 0.25)
ax.indicate_inset_zoom(axins)
ax.set_xlabel('axial depth [pixels]',fontname ='Arial')

aspect = r_log.shape[1]/ r_log.shape[0]
ax = fig.add_subplot(gs[0, 2])
ax.imshow(r_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin)
ax.axvline(x=index,ymax = 1, ymin=0.5, linewidth=2, color='orange', linestyle='--')
ax.set_title('𝜆 = %.3f (PSF convolved): %d-%d' % (lmbda, vmax, rvmin))
ax.set_axis_off()

axins = ax.inset_axes([0.45,0.55,0.52, 0.42])
axins.imshow(r_log,'gray', aspect=aspect, vmax=vmax, vmin=rvmin,origin="lower")
x1, x2, y1, y2 = 2000, 4000, 225, 145
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
ax.indicate_inset_zoom(axins, edgecolor="red")

b_line = abs(r_line[:,index])

ax = fig.add_subplot(gs[1,2])
ax.set_yticks([])
ax.plot(b_line)
ax.set_xlabel('axial depth [pixels]',fontname ='Arial')
axins = ax.inset_axes([0.6, 0.3, 0.37, 0.67])
axins.set_xticks([])
axins.set_yticks([])
axins.plot(b_line)
axins.set_xlim(140, 200)
axins.set_ylim(0, 0.25)

ax.indicate_inset_zoom(axins)

ax = fig.add_subplot(gs[:, 3])
ax.plot(abs(D))

ax.set_xlabel('axial depth [pixels]',fontname ='Arial')
ax.set_ylabel('normalized intensity [a.u.]',fontname ='Arial')
axins = ax.inset_axes([0.6, 0.3, 0.37, 0.3])
axins.set_xticks([])
axins.set_yticks([])
axins.plot(abs(D))
axins.set_xlim(140, 200)
axins.set_ylim(0, 0.06)
ax.set_title('estimated PSF')
ax.indicate_inset_zoom(axins)

plt.show()


fig,ax = plt.subplots(4,3,figsize = (18,13),constrained_layout=True)
ax[0,0].plot(s_line)
ax[0,0].axhline(y = 0.05, color= 'red', linestyle = '--')

ax[0,0].set_title('reference line (0.05)')
ax[0,1].plot(x_line)
ax[0,1].set_title('sparse line')
ax[0,1].set_yticks([])
ax[0,2].plot(b_line)
ax[0,2].axhline(y =0.05, color= 'red', linestyle = '--')

ax[0,2].set_yticks([])
ax[0,2].set_title('convolved line')

ax[1,0].plot(abs(s[:,index]))
ax[1,0].set_title('reference line(l2 scaled) (10,000)')
ax[1,0].axhline(y = 10000, color= 'red', linestyle = '--')
ax[1,1].plot(abs(x[:,index]))
ax[1,1].set_yticks([])
ax[1,1].set_title('sparse line(l2 scaled)')
ax[1,2].plot(abs(r[:,index]))
ax[1,2].axhline(y = 10000, color= 'red', linestyle = '--')
ax[1,2].set_yticks([])
ax[1,2].set_title('convolved line(l2 scaled)')

ax[2,0].plot(20*np.log10(abs(s[:,index])))
ax[2,0].set_title('reference line(20 log)(80)')
ax[2,0].axhline(y = 80, color= 'red', linestyle = '--')

ax[2,1].plot(20*np.log10(abs(x[:,index])))

ax[2,1].set_yticks([])
ax[2,1].set_title('sparse line(20 log)')

ax[2,2].plot(20*np.log10(abs(r[:,index])))
ax[2,2].set_yticks([])
ax[2,2].axhline(y = 80, color= 'red', linestyle = '--')

ax[2,2].set_title('convolved line(20 log)')

ax[3,0].plot(intensity_norm(20*np.log10(abs(s[:,index]))))
ax[3,0].set_title('reference line(intensity_norm)(140)')
ax[3,0].axhline(y = 140, color= 'red', linestyle = '--')
ax[3,1].plot(intensity_norm(20*np.log10(abs(x[:,index]))))
ax[3,1].set_title('sparse line(intensity_norm)')
ax[3,1].set_yticks([])
ax[3,2].plot(intensity_norm(20*np.log10(abs(r[:,index]))))
ax[3,2].set_title('convolved line(intensity_norm)')
ax[3,2].axhline(y = 140, color= 'red', linestyle = '--')
ax[3,2].set_yticks([])

plt.show()

# vmin = 160
# rm = np.where(s_log_norm <= vmin, 0, s_log_norm)
# sm = np.where(sparse <= vmin, 0, sparse)
#
# image = Image.fromarray(rm.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/original.bmp')
#
# image = Image.fromarray(sm.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/sparse.bmp')
#
#
# image = Image.fromarray(s_log_norm.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/original.bmp')
#
# image = Image.fromarray(sparse.astype(np.uint8))
# out = image.resize((512,330))
# out.save('/Users/youngwang/Desktop/sparse.bmp')
#
#
# # fig, ax = plt.subplots(1,2, figsize = (16,9), constrained_layout=True)
# fig, ax = plt.subplots(1,2, figsize = (16,9))
#
# rm = np.array(Image.open('/Users/youngwang/Desktop/original.bmp'))
# sm = np.array(Image.open('/Users/youngwang/Desktop/sparse.bmp'))
# ax[0].imshow(rm,'gray', vmax=vmax, vmin=140)
# ax[0].set_title('reference %d-%d' % (vmax, 140))
# ax[0].set_axis_off()
#
# vmin = 20
# ax[1].imshow(sm,'gray', vmax=vmax, vmin=vmin)
# ax[1].set_title('𝜆 = %.3f %d-%d' % (lmbda, vmax, vmin))
# ax[1].set_axis_off()
# plt.tight_layout()
# plt.show()
