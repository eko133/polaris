# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv

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
    values = data['z'].values
    x_range = data.x.max() - data.x.min()
    y_range = data.y.max() - data.y.min()
    grid_x, grid_y = np.mgrid[data.x.min():data.x.max():(x_range * 1j), data.y.min():data.y.max():(y_range * 1j)]
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear').astype(np.uint8)
    grid_z0 = grid_z0.T
    im = Image.fromarray(grid_z0, 'L')
    plt.imshow(im)
    return im.show()


def extract_coordinates(data, cor = 'index'):
    """
    extract coordinates from series/index
    """
    if cor == 'index':
        data['x'] = data.index.str.extract(r'R00X(.*?)Y').astype(int).values
        data['y'] = data.index.str.extract(r'Y(.*?)$').astype(int).values
    else:
        data['x'] = data[cor].values.str.extract(r'R00X(.*?)Y').astype(int)
        data['y'] = data[cor].values.str.extract(r'Y(.*?)$').astype(int)
    return data['x'].values, data['y'].values


def coordinates_transform(x,y,data='gdgt'):
    if data == 'gdgt':
        a = 4.676282051
        b = -260.2083333
        c = -5.163934426
        d = 532.7377049
    elif data == 'sterol':
        a = 4.660282258
        b = -377.9351478
        c = -4.614285714
        d = 461.5142857
    return c*y+d, a*x+b


def image_warping(src_image,src_pt,dst_pt):
    src_pt = np.array([[35,147],[50,315],[283,336],[306,217]]).astype(np.float32)
    dst_pt = np.array([[35,147],[50,315],[283,148],[306,57]]).astype(np.float32)
    warp_mat = cv.getAffineTransform(src_pt,dst_pt)
    warp_dst = cv.warpAffine(src_image,warp_mat,(src_image.shape[1],src_image.shape[0]))
    return warp_dst


def coordinates_transform(srcTri,dstTri):
    """
    solve the affine transformation matrix between FlexImage coordinates and X_ray coordinates
    https://stackoverflow.com/questions/56166088/how-to-find-affine-transformation-matrix-between-two-sets-of-3d-points
    """
    l = len(srcTri)
    B = np.vstack([np.transpose(srcTri), np.ones(l)])
    D = 1.0 / np.linalg.det(B)
    entry = lambda r, d: np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))
    M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(dstTri)]
    A, t = np.hsplit(np.array(M), [l - 1])
    t = np.transpose(t)[0]
    return A, t



    # if data == 'gdgt':
    #     a = 4.676282051
    #     b = -260.2083333
    #     c = -5.163934426
    #     d = 532.7377049
    # elif data == 'sterol':
    #     a = 4.660282258
    #     b = -377.9351478
    #     c = -4.614285714
    #     d = 461.5142857
    # return c*y+d, a*x+b
