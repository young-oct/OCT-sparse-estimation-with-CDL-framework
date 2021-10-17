# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 7:00 p.m.
# @Author  : young wang
# @FileName: lamba_compare.py
# @Software: PyCharm

'''this script generates images for the figure 3 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
various values of the regularization parameter lambda'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from misc import processing,quality,annotation
from sporco.admm import cbpdn


# Module level constants
eps = 1e-14


def sparse_recon(s, D,lmbda):

    l2f, s_norm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    b = cbpdn.ConvBPDN(D, s_norm, lmbda, opt=opt_par, dimK=1, dimN=1)
    _ = b.solve()
    r_norm = b.reconstruct().squeeze()

    r = processing.from_l2_normed(r_norm, l2f)

    return 20 * np.log10(abs(r))

# np.seterr(divide = 'ignore')

if __name__ == '__main__':

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
    file_name = ['ear']
    # Load the example dataset
    s, D = processing.load_data(file_name[0], decimation_factor=20)
    rvmin, vmax = 5, 55 #dB

    s_log = 20 * np.log10(abs(s))

    # l2 norm data and save the scaling factor
    _, snorm = processing.to_l2_normed(s)

    speckle_weight = 0.1

    lmbda = [1e-5, 0.01, 0.05,0.10,0.15, 0.17]

    w_lmbda = 0.05

    index = 400 # index A-line
    s_line = abs(snorm[:,index])

    x_line = np.zeros((snorm.shape[0], len(lmbda)))
    sparse = np.zeros((snorm.shape[0], snorm.shape[1], len(lmbda)))
    r0_log = np.zeros((snorm.shape[0], snorm.shape[1], len(lmbda)))

    for i in range(len(lmbda)):

        x, line = processing.make_sparse_representation(s, D, lmbda[i],w_lmbda, speckle_weight, Line=True,index = index, Ear=True )
        x_log = 20 * np.log10(abs(x))

        r0_log[:,:,i] = sparse_recon(s, D, lmbda[i])
        sparse[:,:,i] = x_log

        x_line[:, i] = line

    width, height = (115, 95)
    homogeneous = [[125, 120, width, height]]

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(ncols=len(lmbda) + 1, nrows=4)

    aspect = s_log.shape[1] / s_log.shape[0]

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(s_log, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
    ax.set_axis_off()
    ax.set_title('reference')
    ax.axvline(x=index, linewidth=1, color='orange', linestyle='--')

    for k in range(len(homogeneous)):
        for j in annotation.get_homogeneous(*homogeneous[k]):
            ax.add_patch(j)

    ho_original = quality.ROI(*homogeneous[0], s_log)

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(ho_original, 'gray', aspect=ho_original.shape[1] / ho_original.shape[0], vmax=vmax, vmin=rvmin,interpolation='none')
    ax.set_axis_off()
    ax.annotate('', xy=(72.5, 10), xycoords='data',
                xytext=(60, 5), textcoords='data',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    circ = patches.Circle((80, 55), 15, alpha=1, fill=False,edgecolor = 'red',
                          linestyle='--',transform=ax.transData)
    ax.add_patch(circ)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(s_line)

    ax.set_xlabel('axial depth [pixels]')
    ax.set_ylabel('normalized \nmagnitude [a.u.]',fontsize=14)
    ax.set_ylim(0, np.max(s_line)*1.1)

    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.75, 0.95, 'sparse estimate\nimages',
            rotation=90,
            horizontalalignment='center',
            verticalalignment='top',
            multialignment='center', transform=ax.transAxes, fontsize=15)
    ax.set_axis_off()


    for i in range(len(lmbda)):

        aspect = sparse[:, :, i].shape[1]/sparse[:, :, i].shape[0]

        ax = fig.add_subplot(gs[0, i + 1])

        if i == 0:

            ax.set_title(r'$𝜆$= {:.2e}'.format(lmbda[0]))

        else:
            ax.set_title(r'$𝜆$ = %.2f' %(float(lmbda[i])))


        ax.imshow(r0_log[:, :, i], 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
        ax.set_axis_off()


        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(sparse[:, :, i], 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
        ax.axvline(x=index, linewidth=1, color='orange', linestyle='--')

        if i == 0:
            textstr = '\n'.join((
            r'$𝜆$= {:.2e}'.format(lmbda[0]),
            r'$W$ = %.1f' % (speckle_weight)))

        else:
            textstr = '\n'.join((
                r'$𝜆$ = %.2f ' % (lmbda[i]),
                r'$W$ = %.1f' % (speckle_weight)))
        ax.set_title(textstr)

        ax.set_axis_off()
        for k in range(len(homogeneous)):
            for j in annotation.get_homogeneous(*homogeneous[k]):
                ax.add_patch(j)

        ho_x = quality.ROI(*homogeneous[0],  sparse[:, :, i])

        aspect = width / height
        ax = fig.add_subplot(gs[2, i + 1])
        ax.imshow(ho_x, 'gray', aspect=aspect, vmax=vmax, vmin=rvmin,interpolation='none')
        ax.annotate('', xy=(72.5, 10), xycoords='data',
                    xytext=(60, 5), textcoords='data',
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        circ = patches.Circle((80, 55), 15, alpha=1, fill=False, edgecolor='red',
                              linestyle='--', transform=ax.transData)
        ax.add_patch(circ)
        ax.set_axis_off()

        ax = fig.add_subplot(gs[3, i + 1])
        ax.plot(x_line[:, i])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(s_line)*1.1)

        ax.set_xlabel('axial depth [pixels]')
    plt.show()

    fig.savefig('../Images/lambda_compare.svg',
                dpi = 1200,
                transparent=True,format = 'svg')

