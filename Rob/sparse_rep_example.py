# -*- coding: utf-8 -*-
# @Time    : 2021-04-07 11:57 p.m.
# @Author  : young wang
# @FileName: res_finger.py
# @Software: PyCharm


'''this script shows an indirect evidence that the sparse image is super resolution
version of the original image through 1d convolution'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sporco import prox
from sporco.admm import cbpdn
from skimage.exposure import match_histograms
from skimage.filters import gaussian
    
#Module level constants
eps = 1e-14

def imag2uint(data,vmin,vmax):
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin),0))
    return pixel_vals


def gaussian_blur(noisy, sigma=0.5): 
    out = gaussian(noisy, sigma=sigma, output=None, mode='nearest', cval=0, 
             multichannel=None, preserve_range=False, truncate=4.0)
    return(out)
def load_data(dataset_name, decimation_factor=1):
    # define signal & dictionary path
    if dataset_name == 'finger':
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
        
        #Only  keep every 30th line
        s=s[:,::30]
        return(s, D)
    else:
        raise Exception("Dataset %s not found" % dataset_name)

def plot_images(plot_titles, plot_data, 
                vmin, vmax, overlay_plots=None,
                suptitle=None):
    assert len(plot_titles)==len(plot_data), 'Length of plot_titles does not match length of plot_data'
    
    nplots=len(plot_titles)
    fig, axes = plt.subplots(1, nplots)
    if overlay_plots is None:
        overlay_plots=[None for i in range(len(plot_titles))]

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im, overlay_plot) in enumerate(zip(axes, plot_titles, plot_data, overlay_plots)):
        ax.set_title(title)
        ax.imshow(im,vmax=vmax,vmin=rvmin,cmap='gray')
        if overlay_plot is not None:
            ax.plot(overlay_plot)

def plot_alines(plot_titles, plot_data, 
                vmin, vmax,
                suptitle=None):
    assert len(plot_titles)==len(plot_data), 'Length of plot_titles does not match length of plot_data'
    
    nplots=len(plot_titles)
    fig, axes = plt.subplots(1, nplots)
    
    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes, plot_titles, plot_data)):
        ax.set_title(title)
        ax.plot(im, lw=0.5)
        ax.set_ylim(vmin,vmax)

def to_l2_normed(s):
    l2f = prox.norm_l2(s,axis=0).squeeze()
    return(l2f, s / l2f)
    
def from_l2_normed(s, l2f):
    return(s * l2f)


if __name__ == '__main__':

    plt.close('all')        
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 12
            #'text.usetex': False,
            #'font.family': 'stixgeneral',
            #'mathtext.fontset': 'stix',
        }
    )
    
    #Load the example dataset
    s, D = load_data('finger', decimation_factor=30)
    # (3) l2 norm data and save the scaling factor
    l2f, snorm = to_l2_normed(s)
    
    #Define CBPN solver params
    opt_par = cbpdn.ConvBPDNMaskDcpl.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})    
    # define image params
    lmbda = 0.01
    rvmin = 60 #dB
    vmax = 110 #dB
    #Weigth factor to apply to the fidelity (l2) term in the cost function
    #in regions segmented as containing speckle
    speckle_weight=5
    
    
    
    W=np.ones_like(s)
    #First pass with unit weighting - same as ConvBPDN
    b=cbpdn.ConvBPDNMaskDcpl(D, snorm, lmbda, W, opt=opt_par, dimK=1, dimN=1)
    #Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    #Caclulate sparse reconstruction
    rnorm = b.reconstruct().squeeze()
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)
    #Dictionary PSF
    Dout=b.D.squeeze().shape
    # Convert back from normalized 
    s = from_l2_normed(snorm, l2f)
    x = from_l2_normed(xnorm, l2f)
    r = from_l2_normed(rnorm, l2f)

    s_log = 20 * np.log10(abs(s))            
    x_log = 20 * np.log10(abs(x))
    r_log = 20 * np.log10(abs(r))
    
    match = match_histograms(x_log,r_log,multichannel=False)
    match_log = np.where(match <= np.min(match),0,match)

    #Calculate a contour of the main reflector, the one that needs to be 
    #deconvolved
    from scipy.signal import filtfilt, firwin
    b=firwin(35,0.08)
    #Below this contour we will apply the speckle_weight
    contour1 = np.argmax(gaussian_blur(r_log), axis=0)+10
    contour1 = filtfilt(b,1,contour1)

    #%%
    plot_images(['Reference', 'Sparse rep', 
                 'Sparse vectors', 'Sparse vectors (matched)'],
                [s_log, r_log, x_log, match_log],
                rvmin, vmax, 
                overlay_plots=[contour1, None, None, None],
                suptitle='Pass 1')
    #%%    
    plot_alines(['Reference', 'Sparse rep', 
                 'Sparse vectors', 'Sparse vectors (matched)'],
                [s_log, r_log, x_log, match_log],
                rvmin, vmax, suptitle='Pass 1')
    
    #Second pass after segmentation
    W=np.ones_like(s_log)
    for i in range(W.shape[1]):
        W[np.int32(contour1[i]):,i]=speckle_weight
        
    b1=cbpdn.ConvBPDNMaskDcpl(D, snorm, lmbda, W, opt=opt_par, dimK=1, dimN=1)
    #Second time!
    x1norm = b1.solve().squeeze()+eps
    r1norm = b1.reconstruct().squeeze()
    x1norm = np.roll(x1norm, np.argmax(D), axis=0)

    x1 = from_l2_normed(x1norm, l2f)
    r1 = from_l2_normed(r1norm, l2f)
    
    x1_log = 20 * np.log10(abs(x1))
    r1_log = 20 * np.log10(abs(r1))
    
    x1_log_blur = gaussian_blur(x1_log, sigma=0.3)
    
    #%%
    plot_images(['Reference', 'Sparse rep', 
                 'Sparse vectors', 'Sparse vectors (w/ blur)'],
                [s_log, r1_log, x1_log, x1_log_blur],
                rvmin, vmax,
                suptitle='Pass 2')
    #%%    
    plot_alines(['Reference', 'Sparse rep', 
                 'Sparse vectors', 'Sparse vectors (w/ blur)'],
                [s_log, r1_log, x1_log, x1_log_blur],
                rvmin, vmax, suptitle='Pass 2')