# -*- coding: utf-8 -*-
# @Time    : 2021-04-07 11:57 p.m.
# @Author  : young wang
# @FileName: res_finger.py
# @Software: PyCharm


'''this script shows an indirect evidence that the sparse image is super resolution
version of the original image through 1d convolution'''

import sys
import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sporco import prox
import copy
from sporco.admm import cbpdn
from skimage.exposure import match_histograms

eps = 1e-14

def imag2uint(data,vmin,vmax):
    pixel_vals = round(255 * (data - vmin) / (vmax - vmin),0)
    return pixel_vals

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
S_PATH = './finger'
D_PATH = './finger_1.05'

# load signal & dictionary
with open(S_PATH, 'rb') as f:
    s = pickle.load(f).T
    f.close()

with open(D_PATH, 'rb') as f:
    D = pickle.load(f)
    f.close()

# (2) remove background noise: minus the frame mean
s = s - np.mean(s, axis=1)[:, np.newaxis]

# (3) l2 norm data and save the scaling factor
l2f = prox.norm_l2(s,axis=0).squeeze()
s = s / l2f

Maxiter = 20
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                  'MaxMainIter': Maxiter, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                  'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

# define a test line
index = 4000
lmbda = 0.03
svmin = 70
rvmin = 140

b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt_par, dimK=1, dimN=1)
x = b.solve()
x = x.squeeze()
r = b.reconstruct().squeeze()
x = np.roll(x, np.argmax(D), axis=0)
#Keep the log finite
x = x + eps
#Dictionary PSF
Dout=b.D.squeeze().shape

s = s * l2f
x = x * l2f
r = r * l2f
        
x_log = 20 * np.log10(abs(x))
r_log = 20 * np.log10(abs(r))
s_log = 20 * np.log10(abs(s))

match = match_histograms(x_log,r_log,multichannel=False)
sparse = np.where(match <= np.min(match),0,match)

#%%
plt.figure()
plt.subplot(141)
plt.plot(s_log[:,::30])
plt.ylim([60,120])
plt.title('Reference')
plt.subplot(142)
plt.title('Convolution')
plt.plot(r_log[:,::30])
plt.ylim([60,120])
plt.subplot(143)
plt.title('Sparse')
plt.plot(x_log.T[:,::30])
plt.ylim([60,120])
plt.subplot(144)
plt.title('Sparse (matched)')
plt.plot(sparse[:,::30])
plt.ylim([60,120])

#%%
index=190
plt.figure()
plt.plot(s_log[:,index],label='s')
plt.plot(r_log[:,index],label='r')
plt.plot(sparse[:,index], label='hist-match')
plt.plot(x_log[:,index], label='x')
#plt.ylim([np.amin(s_log[:,index]),np.amax(s_log[:,index])])
plt.legend()



from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.restoration import denoise_nl_means, denoise_bilateral
from skimage.filters import gaussian
from scipy.ndimage import median_filter
# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(s_log[0:50,0:50], average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.

#scipy.ndimage.median_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0
s1=s_log[:,::30]
r1=r_log[:,::30]
sp1=x_log[:,::30]
sp2=sparse[:,::30]


rvmin=65
vmax=110


bayes = lambda noisy: denoise_wavelet(noisy, multichannel=True, 
                            method='BayesShrink', mode='soft',
                            rescale_sigma=True)
visushrink=lambda noisy: denoise_wavelet(noisy, multichannel=True, 
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est/4, rescale_sigma=True)

nl_means=lambda noisy: denoise_nl_means(noisy, patch_size=7, patch_distance=11, h=0.1)

bilateral = lambda noisy: denoise_bilateral(noisy, win_size=None, sigma_color=None, 
                                           sigma_spatial=1, bins=10000, 
                                           mode='constant', cval=0, 
                                           multichannel=False)
no_denoiser = lambda noisy: noisy

gaussian_blur = lambda noisy: gaussian(noisy, sigma=0.55, output=None, mode='nearest', cval=0, 
                                       multichannel=None, preserve_range=False, truncate=4.0)


denoiser=gaussian_blur

plt.figure()
plt.subplot(141)
plt.imshow(s1,vmax=vmax,vmin=rvmin,cmap='gray')
plt.title('Reference')
plt.subplot(142)
plt.imshow(r1,vmax=vmax,vmin=rvmin,cmap='gray')
plt.title('Convolved')
plt.subplot(143)
plt.title('Sparse vectors')
plt.imshow(sp1,cmap='gray')
plt.subplot(144)
plt.title('Histogram-matched')
plt.imshow(denoiser(sp2),vmax=vmax,vmin=rvmin,cmap='gray')





# rvmin=65
# vmax=110
# plt.figure()
# plt.subplot(141)
# plt.imshow(wavelet_denoiser(s_log_norm)[:,::30],vmax=vmax,vmin=rvmin,cmap='gray')
# plt.title('Reference')
# plt.subplot(142)
# plt.imshow(wavelet_denoiser(r_log)[:,::30],vmax=vmax,vmin=rvmin,cmap='gray')
# plt.title('Convolved')
# plt.subplot(143)
# plt.title('Sparse vectors')
# rvmin=60
# vmax=110
# plt.imshow(wavelet_denoiser(x_log.T)[:,::30],vmax=vmax,vmin=rvmin,cmap='gray')
# plt.subplot(144)
# plt.title('Histogram-matched')
# plt.imshow(wavelet_denoiser(sparse)[::30],vmax=vmax,vmin=rvmin,cmap='gray')



#%%

# axins = ax.inset_axes([0.45,0.55,0.52, 0.42])
# axins.imshow(r_log,'gray', aspect=aspect, vmax=vmax, vmin=rvmin,origin="lower")
# x1, x2, y1, y2 = 2000, 4000, 225, 145
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticklabels('')
# axins.set_yticklabels('')
# ax.indicate_inset_zoom(axins, edgecolor="red")

# b_line = abs(r_line[:,index])

# ax = fig.add_subplot(gs[1,2])
# ax.set_yticks([])
# ax.plot(b_line)
# ax.set_xlabel('axial depth [pixels]',fontname ='Arial')
# axins = ax.inset_axes([0.6, 0.3, 0.37, 0.67])
# axins.set_xticks([])
# axins.set_yticks([])
# axins.plot(b_line)
# axins.set_xlim(140, 200)
# axins.set_ylim(0, 0.25)

# ax.indicate_inset_zoom(axins)

# ax = fig.add_subplot(gs[:, 3])
# ax.plot(abs(D))

# ax.set_xlabel('axial depth [pixels]',fontname ='Arial')
# ax.set_ylabel('normalized intensity [a.u.]',fontname ='Arial')
# axins = ax.inset_axes([0.6, 0.3, 0.37, 0.3])
# axins.set_xticks([])
# axins.set_yticks([])
# axins.plot(abs(D))
# axins.set_xlim(140, 200)
# axins.set_ylim(0, 0.06)
# ax.set_title('estimated PSF')
# ax.indicate_inset_zoom(axins)

# plt.show()


# fig,ax = plt.subplots(4,3,figsize = (18,13),constrained_layout=True)
# ax[0,0].plot(s_line)
# ax[0,0].axhline(y = 0.05, color= 'red', linestyle = '--')

# ax[0,0].set_title('reference line (0.05)')
# ax[0,1].plot(x_line)
# ax[0,1].set_title('sparse line')
# ax[0,1].set_yticks([])
# ax[0,2].plot(b_line)
# ax[0,2].axhline(y =0.05, color= 'red', linestyle = '--')

# ax[0,2].set_yticks([])
# ax[0,2].set_title('convolved line')

# ax[1,0].plot(abs(s[:,index]))
# ax[1,0].plot(abs(r[:,index]))
# ax[1,0].set_title('reference line(l2 scaled) (10,000)')
# ax[1,0].axhline(y = 10000, color= 'red', linestyle = '--')
# ax[1,1].plot(abs(x[:,index]))
# ax[1,1].set_yticks([])
# ax[1,1].set_title('sparse line(l2 scaled)')
# ax[1,2].plot(abs(r[:,index]))
# ax[1,2].axhline(y = 10000, color= 'red', linestyle = '--')
# ax[1,2].set_yticks([])
# ax[1,2].set_title('convolved line(l2 scaled)')

# ax[2,0].plot(20*np.log10(abs(s[:,index])))
# ax[2,0].set_title('reference line(20 log)(80)')
# ax[2,0].axhline(y = 80, color= 'red', linestyle = '--')

# ax[2,1].plot(20*np.log10(abs(x[:,index])))

# ax[2,1].set_yticks([])
# ax[2,1].set_title('sparse line(20 log)')

# ax[2,2].plot(20*np.log10(abs(r[:,index])))
# ax[2,2].set_yticks([])
# ax[2,2].axhline(y = 80, color= 'red', linestyle = '--')

# ax[2,2].set_title('convolved line(20 log)')

# ax[3,0].plot(intensity_norm(20*np.log10(abs(s[:,index]))))
# ax[3,0].set_title('reference line(intensity_norm)(140)')
# ax[3,0].axhline(y = 140, color= 'red', linestyle = '--')
# ax[3,1].plot(intensity_norm(20*np.log10(abs(x[:,index]))))
# ax[3,1].set_title('sparse line(intensity_norm)')
# ax[3,1].set_yticks([])
# ax[3,2].plot(intensity_norm(20*np.log10(abs(r[:,index]))))
# ax[3,2].set_title('convolved line(intensity_norm)')
# ax[3,2].axhline(y = 140, color= 'red', linestyle = '--')
# ax[3,2].set_yticks([])

# plt.show()

# # vmin = 160
# # rm = np.where(s_log_norm <= vmin, 0, s_log_norm)
# # sm = np.where(sparse <= vmin, 0, sparse)
# #
# # image = Image.fromarray(rm.astype(np.uint8))
# # out = image.resize((512,330))
# # out.save('/Users/youngwang/Desktop/original.bmp')
# #
# # image = Image.fromarray(sm.astype(np.uint8))
# # out = image.resize((512,330))
# # out.save('/Users/youngwang/Desktop/sparse.bmp')
# #
# #
# # image = Image.fromarray(s_log_norm.astype(np.uint8))
# # out = image.resize((512,330))
# # out.save('/Users/youngwang/Desktop/original.bmp')
# #
# # image = Image.fromarray(sparse.astype(np.uint8))
# # out = image.resize((512,330))
# # out.save('/Users/youngwang/Desktop/sparse.bmp')
# #
# #
# # # fig, ax = plt.subplots(1,2, figsize = (16,9), constrained_layout=True)
# # fig, ax = plt.subplots(1,2, figsize = (16,9))
# #
# # rm = np.array(Image.open('/Users/youngwang/Desktop/original.bmp'))
# # sm = np.array(Image.open('/Users/youngwang/Desktop/sparse.bmp'))
# # ax[0].imshow(rm,'gray', vmax=vmax, vmin=140)
# # ax[0].set_title('reference %d-%d' % (vmax, 140))
# # ax[0].set_axis_off()
# #
# # vmin = 20
# # ax[1].imshow(sm,'gray', vmax=vmax, vmin=vmin)
# # ax[1].set_title('𝜆 = %.3f %d-%d' % (lmbda, vmax, vmin))
# # ax[1].set_axis_off()
# # plt.tight_layout()
# # plt.show()
