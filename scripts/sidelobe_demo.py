# -*- coding: utf-8 -*-
# @Time    : 2021-10-06 2:12 p.m.
# @Author  : young wang
# @FileName: sidelobe_deom.py
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

    fig,ax = plt.subplots(2,2, figsize=(16,9), constrained_layout = True)
    finger = plt.imread('../Images/sidelobe_finger.jpeg')
    onion = plt.imread('../Images/sidelobe_onion.jpeg')
    ear = plt.imread('../Images/sidelobe_ear.jpeg')
    ear2 = plt.imread('../Images/sidelobe_ear2.jpeg')


    ax[0,0].imshow(finger)

    ax[0,0].set_axis_off()

    ax[0,1].imshow(onion)

    ax[0,1].set_axis_off()
    ax[1,0].imshow(ear)

    ax[1,0].set_axis_off()

    ax[1,1].set_axis_off()
    ax[1,1].imshow(ear2)

    ax[1,1].set_axis_off()

    plt.show()

    fig.savefig('../Images/sidelobe_deom.svg',
                dpi = 1200,
                transparent=True,format = 'svg')