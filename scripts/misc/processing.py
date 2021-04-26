# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 3:54 p.m.
# @Author  : young wang
# @FileName: processing.py
# @Software: PyCharm
import numpy as np
from sporco import prox
import pickle
from pathlib import Path


def imag2uint(data, vmin, vmax):
    data=np.clip(data,vmin,vmax)
    pixel_vals = np.uint8(np.around(255 * (data - vmin) / (vmax - vmin), 0))
    return pixel_vals

def to_l2_normed(s):
    l2f = prox.norm_l2(s, axis=0).squeeze()
    return (l2f, s / l2f)

def from_l2_normed(s, l2f):
    return (s * l2f)

def load_data(dataset_name, decimation_factor):
    # check if such file exists
    S_PATH = '../Data/' + dataset_name
    D_PATH = '../Data/PSF/' + dataset_name
    if Path(S_PATH).is_file() and Path(D_PATH).is_file():
        #load data & dictionary
        with open(S_PATH, 'rb') as f:
            s = pickle.load(f).T
            f.close()

        with open(D_PATH, 'rb') as f:
            D = pickle.load(f)
            f.close()

        # (2) remove background noise: minus the frame mean
        s = s - np.mean(s, axis=1)[:, np.newaxis]
        # (3) sample every decimation_factor line,
        s = s[:, ::decimation_factor]
        return (s, D)
    else:
        raise Exception("Dataset %s not found" % dataset_name)




