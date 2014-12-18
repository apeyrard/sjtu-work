#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import argparse
import scipy.signal as ss
import scipy.ndimage.filters as filters
import math

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data, dtype=float).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

def roberts(matrix):
    gx = np.zeros(matrix.shape)
    gy = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            try:
                gx[x][y] = matrix[x+1][y+1] - matrix[x][y]
                gy[x][y] = matrix[x+1][y] - matrix[x][y+1]
            except IndexError:
                gx[x][y] = 0
                gy[x][y] = 0
    return gx, gy

def prewitt(matrix):
    gx = np.zeros(matrix.shape)
    gy = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            try:
                gx[x][y] = (matrix[x+1][y-1]
                                        + matrix[x+1][y]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - matrix[x-1][y]
                                        - matrix[x-1][y+1])
                gy[x][y] = (matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - matrix[x][y-1]
                                        - matrix[x+1][y-1])
            except IndexError:
                gx[x][y] = 0
                gy[x][y] = 0
    return gx, gy

def sobel(matrix):
    gx = np.zeros(matrix.shape)
    gy = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            try:
                gx[x][y] = (matrix[x+1][y-1]
                                        + 2*matrix[x+1][y]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - 2*matrix[x-1][y]
                                        - matrix[x-1][y+1])
                gy[x][y] = (matrix[x-1][y+1]
                                        + 2*matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - 2*matrix[x][y-1]
                                        - matrix[x+1][y-1])
            except IndexError:
                gx[x][y] = 0
                gy[x][y] = 0
    return gx, gy

def prewitt_diagonal(matrix):
    gx = np.zeros(matrix.shape)
    gy = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            try:
                gx[x][y] = (matrix[x-1][y]
                                        + matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        - matrix[x][y-1]
                                        - matrix[x+1][y-1]
                                        - matrix[x+1][y])
                gy[x][y] = (matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        + matrix[x+1][y]
                                        - matrix[x][y-1]
                                        - matrix[x-1][y-1]
                                        - matrix[x-1][y])
            except IndexError:
                gx[x][y] = 0
                gy[x][y] = 0
    return gx, gy

def sobel_diagonal(matrix):
    gx = np.zeros(matrix.shape)
    gy = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            try:
                gx[x][y] = (matrix[x-1][y]
                                        + 2*matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        - matrix[x][y-1]
                                        - 2*matrix[x+1][y-1]
                                        - matrix[x+1][y])
                gy[x][y] = (matrix[x][y+1]
                                        + 2*matrix[x+1][y+1]
                                        + matrix[x+1][y]
                                        - matrix[x][y-1]
                                        - 2*matrix[x-1][y-1]
                                        - matrix[x-1][y])
            except IndexError:
                gx[x][y] = 0
                gy[x][y] = 0
    return gx, gy


def detection(matrix, method):
    deltaMat = np.zeros(matrix.shape)
    if method == 'roberts':
        gx, gy = roberts(matrix)
    elif method == 'prewitt':
        gx, gy = prewitt(matrix)
    elif method == 'sobel':
        gx, gy = sobel(matrix)
    elif method == 'prewitt_diagonal':
        gx, gy = prewitt_diagonal(matrix)
    elif method == 'sobel_diagonal':
        gx, gy = sobel_diagonal(matrix)
    elif method == 'mh':
        sigma = 4
        deltaMat = filters.gaussian_filter(matrix, sigma)
        laplacian = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])
        deltaMat = ss.convolve(deltaMat, laplacian, mode='same')
        for x in range(deltaMat.shape[0]):
            for y in range(deltaMat.shape[1]):
                if x ==0 or y==0 or x == deltaMat.shape[0]-1 or y == deltaMat.shape[1]-1:
                    deltaMat[x][y] = 0
        threshold = 4*deltaMat.max()/100

        #searching zero-crossings
        zeroCross = np.zeros(deltaMat.shape)
        for x in range(deltaMat.shape[0]):
            for y in range(deltaMat.shape[1]):
                try:
                    if x-1<0:
                        raise IndexError('Negative Value')
                    a = math.copysign(1, deltaMat[x-1][y])
                    b = math.copysign(1, deltaMat[x+1][y])
                    if a != b and abs(deltaMat[x-1][y] - deltaMat[x+1][y]) > threshold:
                        zeroCross[x][y] = 1
                except IndexError:
                    pass
                try:
                    if x-1<0:
                        raise IndexError('Negative Value')
                    a = math.copysign(1, deltaMat[x][y-1])
                    b = math.copysign(1, deltaMat[x][y+1])
                    if a != b and abs(deltaMat[x][y-1] - deltaMat[x][y+1]) > threshold:
                        zeroCross[x][y] = 1
                except IndexError:
                    pass
                try:
                    if x-1<0:
                        raise IndexError('Negative Value')
                    a = math.copysign(1, deltaMat[x-1][y-1])
                    b = math.copysign(1, deltaMat[x+1][y+1])
                    if a != b and abs(deltaMat[x-1][y-1] - deltaMat[x+1][y+1]) > threshold:
                        zeroCross[x][y] = 1
                except IndexError:
                    pass
                try:
                    if x-1<0:
                        raise IndexError('Negative Value')
                    a = math.copysign(1, deltaMat[x-1][y+1])
                    b = math.copysign(1, deltaMat[x+1][y-1])
                    if a != b and abs(deltaMat[x-1][y+1] - deltaMat[x+1][y-1]) > threshold:
                        zeroCross[x][y] = 1
                except IndexError:
                    pass
        return rescale(zeroCross)
    elif method == 'canny':
        sigma = 4
        deltaMat = rescale(filters.gaussian_filter(matrix, sigma))
        gx, gy = prewitt(deltaMat)
        G = ((gx)**2+(gy)**2)**0.5
        angle = np.zeros(gx.shape, dtype=float)
        g = G.copy()
        for x in range(gx.shape[0]):
            for y in range(gx.shape[1]):
                angle[x][y] = math.degrees(math.atan2(gy[x][y],gx[x][y]))
                if -157.5 <= angle[x][y] < -112.5 or 22.5 <= angle[x][y] < -67.5:
                    angle[x][y] = 45
                    if G[x][y] <= G[x+1][y+1] or G[x][y] <= G[x-1][y-1]:
                        g[x][y] = 0
                elif -112.5 <= angle[x][y] < -67.5 or 67.5 <= angle[x][y] < 112.5:
                    angle[x][y] = 90
                    if G[x][y] <= G[x][y-1] or G[x][y] <= G[x][y+1]:
                        g[x][y] = 0
                elif -67.5 <= angle[x][y] < -22.5 or 112.5 <= angle[x][y] < 157.5:
                    angle[x][y] = 135
                    if G[x][y] <= G[x+1][y-1] or G[x][y] <= G[x-1][y+1]:
                        g[x][y] = 0
                else:
                    angle[x][y] = 0
                    if G[x][y] <= G[x-1][y] or G[x][y] <= G[x+1][y]:
                        g[x][y] = 0
        tl = 7
        th= 2*tl
        gh = np.zeros(gx.shape, dtype=float)
        gl = np.zeros(gx.shape, dtype=float)
        for x in range(gl.shape[0]):
            for y in range(gl.shape[1]):
                if g[x][y] > tl:
                    gl[x][y] = g[x][y]
                if g[x][y] > th:
                    gh[x][y] = g[x][y]
        gl = gl - gh
        valid = np.zeros(gx.shape, dtype=float)
        for x in range(gh.shape[0]):
            for y in range(gh.shape[1]):
                if gh[x][y] > 0:
                    for j in range(-1,2):
                        for k in range(-1,2):
                            try:
                                if x+j<0 or y+k<0:
                                    raise IndexError('Negative value')
                                if gl[x+j][y+j] > 0:
                                    valid[x+j][y+k] = gl[x+j][y+j]
                            except IndexError:
                                pass
        return valid + gh
    return abs(gx) + abs(gy)




parser = argparse.ArgumentParser(description='Image segmentation')

parser.add_argument('image')
parser.add_argument('--roberts', action='store_true')
parser.add_argument('--prewitt', action='store_true')
parser.add_argument('--sobel', action='store_true')
parser.add_argument('--prewitt_diagonal', action='store_true')
parser.add_argument('--sobel_diagonal', action='store_true')
parser.add_argument('--mh', action='store_true')
parser.add_argument('--canny', action='store_true')

args = parser.parse_args()

try:
    with Image.open(args.image) as im:
        matrix = getMatrix(im)

        if args.roberts:
            method = 'roberts'
        elif args.prewitt:
            method = 'prewitt'
        elif args.sobel:
            method = 'sobel'
        elif args.prewitt_diagonal:
            method = 'prewitt_diagonal'
        elif args.sobel_diagonal:
            method = 'sobel_diagonal'
        elif args.mh:
            method = 'mh'
        elif args.canny:
            method = 'canny'

        newMat = detection(matrix, method)

        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



