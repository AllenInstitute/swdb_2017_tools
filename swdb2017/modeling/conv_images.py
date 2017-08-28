import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy import signal

def conv_images(images,filters):
    '''conv_images_dict = conv_images(images,filters)
    conv_image_dict: {filter_index: [list of convoluted images]}
    images: [list of images]
    filters: [list of filters]
    
    This function do convolution on list of images
    with list of filters and return a dictionary of 
    convoluted images "conv_images".
    '''

    conv_images_dict = {}#{'filter_index':'images'}
    for i in range(0,len(filters)):
        temp_imag = []
        for j in range(0,len(images)):
            temp_imag.append(signal.convolve2d(images[j], filters[i], mode='valid'))
    conv_images_dict[i] = temp_imag
    
    return conv_images_dict