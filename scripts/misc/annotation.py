# -*- coding: utf-8 -*-
# @Time    : 2021-04-27 10:17 a.m.
# @Author  : young wang
# @FileName: annotation.py
# @Software: PyCharm

from matplotlib.patches import Rectangle

def get_background(x, y, width, height):
    space = [Rectangle((x, y), width, height, linestyle = '--', linewidth=2, edgecolor='cyan', fill=False)]
    return space

def get_homogeneous(x, y, width, height):
    space = [Rectangle((x, y), width, height,linestyle = '--', linewidth=2, edgecolor='red', fill=False)]
    return space

def get_artifact(x, y, width, height):
    space = [Rectangle((x, y), width, height,linestyle = '--', linewidth=2, edgecolor='green', fill=False)]
    return space