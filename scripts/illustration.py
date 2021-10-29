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
            'font.size': 30,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    fig,ax = plt.subplots(1,2, figsize=(16,9), constrained_layout = True)
    before = plt.imread('../Images/unprocessed_3d.png')
    after = plt.imread('../Images/processed_3d.png')

    ax[0].set_title('(a)')
    ax[0].imshow(before)
    ax[0].annotate('', xy=(380, 180), xycoords='data',
                   xytext=(410, 140), textcoords='data', fontsize=30,
                   color='red', fontname='Arial',
                   arrowprops=dict(facecolor='red', shrink=0.025))
    ax[0].set_axis_off()
    # ax[0].text(x = 5, y = 435, s = 'standard\nprocessing', color = 'white',
    #            weight='bold',
    #            transform = ax[0].transData)
    ax[1].set_title('(b)')

    ax[1].imshow(after)
    # ax[1].text(x = 5, y = 435, s = 'enhanced', color = 'white',
    #            weight='bold',
    #            transform = ax[1].transData)
    ax[1].set_axis_off()

    plt.show()

    fig.savefig('../Images/illustration.jpeg',
                dpi = 800,
                transparent=True,format = 'jpeg')