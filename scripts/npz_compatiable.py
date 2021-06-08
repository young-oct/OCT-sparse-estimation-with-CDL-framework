# @Time    : 2021-06-08 10:12 a.m.
# @Author  : young wang
# @FileName: npz_compatiable.py
# @Software: PyCharm

from misc import processing
import matplotlib.pyplot as plt

from sporco import cnvrep
import matplotlib
import numpy as np
import pickle

from skimage.morphology import disk,square,star,diamond,octagon
from skimage.morphology import dilation, erosion
from skimage import filters
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from sporco.dictlrn import dictlrn
from sporco.admm import cbpdn,ccmod
from sporco import prox

def to_l2_normed(s):
    l2f = prox.norm_l2(s, axis=0).squeeze()
    return (l2f, s / l2f)

def from_l2_normed(s, l2f):
    return (s * l2f)

def plot_images(plot_titles, image,
                vmin, vmax, suptitle=None, overlays=False):
    if overlays != True:
        assert len(plot_titles) == len(image), 'Length of plot_titles does not match length of plot_data'
        image = image
    else:
        assert len(plot_titles) + 1 == len(image)
        mask = image[-1]
        image.pop(-1)

    nplots = len(plot_titles)
    fig, axes = plt.subplots(2, (nplots + 1) // 2, figsize=(18, 13))

    if suptitle is not None:
        plt.suptitle(suptitle)
    for n, (ax, title, im) in enumerate(zip(axes.flatten(), plot_titles, image)):

        if n == 4:
            ax.set_axis_off()
            ax = axes.flatten()[5]

        ax.set_title(title)

        if n != 1:

            ax.set_axis_off()

            if n == 3 and overlays == True:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')
                ax.contour(mask, [0.99], colors='orange', alpha=0.75, linestyles='dashed')

            else:
                ax.imshow(im, aspect=im.shape[1] / im.shape[0], vmax=vmax, vmin=vmin, cmap='gray', interpolation='none')

        else:
            ax.plot(im)
            ax.set_xlabel('Axial depth [pixels]', fontname='Arial')
            ax.set_ylabel('Normalized \nmagnitude [a.u.]', fontname='Arial', fontsize=20)

    plt.tight_layout(pad=0.5)
    plt.show()

def getPSF(s,lmbda):

    l2f, snorm = processing.to_l2_normed(s)

    K = snorm.shape[1]  # number of A-line signal
    M = 1  # state of dictionary

    # randomly select one A-line as the dictionary
    dic_index = int(snorm.shape[1]/2)
    # l2 normalize the dictionary
    D = snorm[:, dic_index]

    # convert to sporco standard layabout
    D = np.reshape(D, (-1, 1, M))

    # uniform random sample the training set from input test, 10%
    train_index = np.random.choice(snorm.shape[1], int(0.25 * K), replace=False)
    s_train = snorm[:, train_index]
    #
    Maxiter = 500

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

    shift = np.argmax(abs(D1)) - 500
    D1 = np.roll(D1, -shift)

    D1 = D1.reshape(-1, 1)

    return D1

def imag2uint(data, vmin, vmax):
    data = np.clip(data, vmin, vmax)
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals

def getWeight(s, D, w_lmbda, speckle_weight,vmax, vmin, Paddging=True, opt_par={}):
    l2f, snorm = to_l2_normed(s)

    b = cbpdn.ConvBPDN(D, snorm, w_lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    x = from_l2_normed(xnorm, l2f)
    x_log = 10 * np.log10(abs(x) ** 2)
    x_log = imag2uint(x_log, vmin, vmax)

    # set thresdhold
    x_log = np.where(x_log <= vmin, 0, x_log)
    W = dilation(x_log, square(4))
    W = erosion(W, square(4))
    W = np.where(W > 0, speckle_weight, 1)

    W = filters.median(W, square(15))

    if Paddging == True:
        pad = 20  #
        # find the bottom edge of the mask with Sobel edge filter

        temp = filters.sobel(W)
        # temp = gaussian_filter(temp, 3)

        pad_value = np.linspace(speckle_weight, 1, pad)

        for i in range(temp.shape[1]):
            peak, _ = find_peaks(temp[:, i], height=0)
            if len(peak) != 0:
                loc = peak[-1]
                if temp.shape[0] - loc >= pad:
                    W[loc:int(loc + pad), i] = pad_value
            else:
                W[:, i] = W[:, i]
    else:
        pass

    W = gaussian_filter(W, sigma=3)
    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W

def make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight,vmax, vmin, Line=False, index=None, Mask=False):
    ''' s -- 2D array of complex A-lines with dims (width, depth)
    '''
    # l2 norm data and save the scaling factor
    l2f, snorm = to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    W = np.roll(getWeight(s, D, w_lmbda, speckle_weight,vmax, vmin, Paddging=True, opt_par=opt_par,), np.argmax(D), axis=0)
    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    xnorm = b.solve().squeeze() + eps
    # calculate sparsity
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    if Line == False and Mask == False:
        ## Convert back from normalized
        x = from_l2_normed(xnorm, l2f)
        return (x)
    elif Line == True and Mask == False:
        assert index != None and 0 <= index <= s.shape[1]
        x = from_l2_normed(xnorm, l2f)
        x_line = abs(xnorm[:, index])
        return x, x_line
    elif Line == False and Mask == True:
        x = from_l2_normed(xnorm, l2f)
        W_mask = np.roll(W, -np.argmax(D), axis=0).squeeze()
        return x, W_mask
    else:
        x = from_l2_normed(xnorm, l2f)
        x_line = abs(xnorm[:, index])
        W_mask = np.roll(W, -np.argmax(D), axis=0).squeeze()
        return x, x_line, W_mask

eps = 1e-14

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    decimation_factor = 10
    rvmin, vmax = 60, 110 #dB

    s = processing.npz_laod('../data/incus.npz','incusImage',decimation_factor)

    # lmbda for dictionary learning step
    d_lmbda = 0.1
    speckle_weight = 0.1

    D = getPSF(s,d_lmbda)
    # with open('/Users/youngwang/Desktop/github/data/PSF/test', 'rb') as f:
    #     D = pickle.load(f)
    #     f.close()

    lmbda = 0.003
    w_lmbda = 0.0015

    l2f, snorm = processing.to_l2_normed(s)

    #
    s_log = 20*np.log10(np.abs(s))

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})
    #
    b0 = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    x0norm = b0.solve().squeeze() + eps
    r0norm = b0.reconstruct().squeeze()

    x0norm = np.roll(x0norm, np.argmax(D), axis=0)
    x0 = from_l2_normed(x0norm, l2f)
    r0 = from_l2_normed(r0norm, l2f)
    #
    x0_log = 20 * np.log10(abs(x0))
    r0_log = 20 * np.log10(abs(r0))

    # update opt to include W
    x1, W = make_sparse_representation(s, D, lmbda,w_lmbda, speckle_weight, vmax =vmax, vmin=rvmin,Mask=True)
    x1_log = 20 * np.log10(abs(x1))

    title = ['(a) reference',
             '(b) Magnitude of the learned PSF $d(z)$',
             '(c) sparse estimation image\n 𝜆 = %.4f' % (lmbda),
             '(d) sparse vector image \nwo/weighting (𝜆 = %.4f)' % (lmbda),
             '(e) sparse vector image \nw/weighting (𝜆 = %.4f,$W$ = %.1f)' % (lmbda, speckle_weight)]

    plot_images(title, [s_log, abs(D), r0_log, x0_log, x1_log, W], rvmin, vmax, overlays=True)



