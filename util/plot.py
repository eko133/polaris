from scipy.interpolate import griddata
from PIL import Image
import numpy as np
import pandas as pd

def enhance_contrast(color,threshold=120):
    """
    enhance contrast of the grayscale x-ray picture
    """

    threshold = threshold
    white = 255
    black = 0
    if color < threshold:
        color = black
    else:
        color = white
    return color


def show_x_ray(data):
    """
    create x-ray image from plain text data matrix
    """
    points = data[['x', 'y']].values
    values = data['color'].values
    x_range = data.x.max() - data.x.min()
    y_range = data.y.max() - data.y.min()
    grid_x, grid_y = np.mgrid[data.x.min():data.x.max():(x_range * 1j), data.y.min():data.y.max():(y_range * 1j)]
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear').astype(np.uint8)
    im = Image.fromarray(grid_z0, 'L')
    return im.show()

