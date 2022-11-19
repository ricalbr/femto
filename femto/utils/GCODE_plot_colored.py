# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:02:29 2022

@author: fedes
"""

import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'
import pandas as pd
import numpy as np


def GCODE_plot_colored(GCODE_array: np.ndarray):
    """
    Function to plot, through plotly, a 3D GCODE, in two colors depending on the shutter parameters
    """
    points_matrix_off = pd.DataFrame(GCODE_array[:, 0:3], columns=["X", "Y", "Z"])
    points_matrix_on = []
    for i in range(len(GCODE_array[:, 4])):
        if GCODE_array[i, 4] == 1:
            points_matrix_on.append(GCODE_array[i, 0:3].tolist())
        else:
            points_matrix_on.append([None, None, None])
            points_matrix_on.append(GCODE_array[i, 0:3].tolist())
    # print(points_matrix_on)
    points_matrix_on_df = pd.DataFrame(points_matrix_on, columns=["X", "Y", "Z"])

    points_matrix_off['shutter'] = 'off'
    points_matrix_on_df['shutter'] = 'on'
    points_matrix = pd.concat([points_matrix_off, points_matrix_on_df])

    fig = px.line_3d(points_matrix, x="X", y="Y", z="Z", color="shutter", title='GCODE Plot',
                     labels={'x': ' X [um]', 'y': 'Y [um]', 'z': 'Z [um]'})
    return fig
