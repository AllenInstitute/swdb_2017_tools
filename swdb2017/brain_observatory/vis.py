"""

This module contains functions that are useful for visualizing Brain
Observatory data.

"""

import matplotlib.pyplot as plt
import h5py
from bokeh.plotting import Figure
from bokeh.models.ranges import Range1d
from bokeh.io import push_notebook, show, output_notebook
import numpy as np
from ipywidgets import interact, Layout, IntSlider, interactive
from bokeh.palettes import Greys256

def get_bokeh_browser(src_data_path):
    """Generates a figure to plot imaging data."""
    height = width = 800
    f_handle = h5py.File(src_data_path)

    data = f_handle['data']
    nrows = data.shape[1]
    ncols = data.shape[2]

    figure = Figure(toolbar_location='left')
    figure.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    figure.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    figure.yaxis.major_label_text_font_size = '0pt'

    figure.x_range = Range1d(0, nrows)
    figure.y_range = Range1d(0, ncols)
    figure.plot_height = height
    figure.plot_width = width
    figure.x_range.bounds = (0, nrows)
    figure.y_range.bounds = (0, ncols)
    figure.toolbar.logo = None

    image = figure.image(image=[np.zeros((nrows, ncols))],
                         x=[0],
                         y=[0],
                         dw=[nrows],
                         dh=[ncols],
                         palette=Greys256)

    def update(change):
        ti = change['new']
        data = f_handle['data']
        img = data[ti, :, :]
        data = {'x': [0], 'y': [0], 'dw': [nrows], 'dh': [ncols],
                'palette': [Greys256], 'image': [img]}
        image.data_source.data = data
        push_notebook()

    slider = IntSlider(min=0,
                       max=data.shape[0],
                       step=data.shape[0]/200,
                       layout=Layout(width='500px'))
    slider.observe(update, names='value')

    return figure, slider
