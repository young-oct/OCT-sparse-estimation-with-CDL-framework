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
            'font.size': 25,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    fig,ax = plt.subplots(3,2, figsize=(16,9), constrained_layout = True)
    finger = plt.imread('../Images/sidelobe/sidelobe_a.jpeg')
    onion = plt.imread('../Images/sidelobe/sidelobe_b.png')
    lens = plt.imread('../Images/sidelobe/sidelobe_c.png')
    skin = plt.imread('../Images/sidelobe/sidelobe_d.png')
    ear = plt.imread('../Images/sidelobe/sidelobe_e.jpeg')
    oral = plt.imread('../Images/sidelobe/sidelobe_f.jpg')

    ax[0,0].imshow(finger)
    ax[0, 0].set_title('(a)')
    ax[0,0].set_axis_off()

    ax[0,0].annotate('', xy=(650, 60), xycoords='data',
                      xytext=(680, 20), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[0,0].annotate('', xy=(680, 70), xycoords='data',
                      xytext=(720, 20), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[0,1].imshow(onion)
    ax[0, 1].set_title('(b)')
    ax[0,1].set_axis_off()

    ax[0, 1].annotate('', xy=(380, 30), xycoords='data',
                      xytext=(420, 5), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[0, 1].annotate('', xy=(410, 110), xycoords='data',
                      xytext=(445, 70), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[1,0].imshow(ear)
    ax[1, 0].set_title('(c)')
    ax[1,0].set_axis_off()

    ax[1, 0].annotate('', xy=(500, 220), xycoords='data',
                      xytext=(540, 160), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )


    ax[1,1].imshow(oral)
    ax[1, 1].set_title('(d)')
    ax[1,1].set_axis_off()
    ax[1, 1].annotate('', xy=(1300, 200), xycoords='data',
                      xytext=(1500, 100), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[2,0].imshow(lens)
    ax[2,0].set_axis_off()
    ax[2, 0].set_title('(e)')
    ax[2, 0].annotate('', xy=(200, 20), xycoords='data',
                      xytext=(240, 5), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[2,1].set_axis_off()
    ax[2,1].imshow(skin)
    ax[2, 1].set_title('(f)')
    ax[2, 1].annotate('', xy=(400, 80), xycoords='data',
                      xytext=(450, 20), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )

    ax[2, 1].annotate('', xy=(350, 80), xycoords='data',
                      xytext=(400, 20), textcoords='data',
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      horizontalalignment='right', verticalalignment='top',
                      )



    plt.show()

    fig.savefig('../Images/sidelobe_deom.jpeg',
                dpi = 800,
                transparent=True,format = 'jpeg')