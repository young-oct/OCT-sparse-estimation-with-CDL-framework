# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 7:52 p.m.
# @Author  : young wang
# @FileName: quality.py
# @Software: PyCharm

def ROI(x, y, width, height,s):
    '''obtain the ROI from the standard layout [330x10240]

    parameters
    ----------
    s has the standard dimension [330x10240]
    y is defined as the axial direction: 330
    x is defined as the lateral direction: 10240
    height refers the increment in the axial direction > 0
    width refers the increment in the lateral direction > 0
    '''
    # fetch ROI
    if height > 0 and width > 0:
        if (x >= 0) and (y >= 0) and (x + width <= s.shape[1]) and (y + height <= s.shape[0]):
            roi = s[y:y + height, x:x + width]
    return roi