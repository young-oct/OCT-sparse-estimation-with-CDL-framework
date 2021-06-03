# -*- coding: utf-8 -*-
# @Time    : 2021-05-25 5:18 p.m.
# @Author  : young wang
# @FileName: convolution_demo.py
# @Software: PyCharm

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

N = 330

psf = signal.gaussian(N, std=5)
structure = np.zeros(N)
structure[100]=0.5
structure[150]=0.3

fig,ax = plt.subplots(1,3,figsize=(25,9))

signal = np.convolve(psf,structure,mode ='same')
ax[0].plot(structure)
ax[0].set_title('structure',fontsize = 40)

ax[1].plot(psf)
ax[1].set_title('PSF',fontsize = 40)
ax[2].plot(signal)
ax[2].set_title('noiseless signal', fontsize = 40)

plt.tight_layout()
plt.show()

fig,ax = plt.subplots(1,4,figsize=(25,9))

signal = np.convolve(psf,structure,mode ='same')
ax[0].plot(structure)
ax[0].set_title('structure',fontsize = 40)

ax[1].plot(psf)
ax[1].set_title('PSF',fontsize = 40)
ax[2].plot(signal)
ax[2].set_title('noiseless signal', fontsize = 40)

noise = np.random.normal(0,0.002,N)
signal += noise
ax[3].plot(signal)
ax[3].set_title('noisy signal',fontsize = 40)
plt.tight_layout()
plt.show()