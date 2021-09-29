# -*- coding: utf-8 -*-
# @Time    : 2021-09-29 1:58 p.m.
# @Author  : young wang
# @FileName: illustration.py
# @Software: PyCharm

import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 23,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    fig,ax = plt.subplots(1,2, figsize=(16,9))
    before = plt.imread('../Images/unprocessed_3d.png')
    after = plt.imread('../Images/processed_3d.png')
    ax[0].imshow(before)
    ax[0].set_axis_off()
    ax[0].set_title('middle ear 3D image(unprocessed)' ,weight='bold')
    ax[1].imshow(after)
    ax[1].set_title('middle ear 3D image(processed)', weight='bold')
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()

    fig.savefig('../Images/illustration.svg',
                dpi = 1200,
                transparent=True,format = 'svg')