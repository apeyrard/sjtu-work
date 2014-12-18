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

def detection(matrix, method):
    deltaMat = np.zeros(matrix.shape)
    if method == 'roberts':
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                try:
                    deltaMat[x][y] = abs(matrix[x+1][y+1] - matrix[x][y]) + abs(matrix[x+1][y] - matrix[x][y+1])
                except IndexError:
                    deltaMat[x][y] = 0
    elif method == 'prewitt':
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                try:
                    if(x-1 < 0 or y-1 < 0):
                        raise IndexError('Negative value')
                    deltaMat[x][y] = (abs(matrix[x+1][y-1]
                                        + matrix[x+1][y]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - matrix[x-1][y]
                                        - matrix[x-1][y+1])
                                        + abs(matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - matrix[x][y-1]
                                        - matrix[x+1][y-1]))

                except IndexError:
                    deltaMat[x][y] = 0
    elif method == 'sobel':
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                try:
                    if(x-1 < 0 or y-1 < 0):
                        raise IndexError('Negative value')
                    deltaMat[x][y] = (abs(matrix[x+1][y-1]
                                        + 2*matrix[x+1][y]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - 2*matrix[x-1][y]
                                        - matrix[x-1][y+1])
                                        + abs(matrix[x-1][y+1]
                                        + 2*matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        - matrix[x-1][y-1]
                                        - 2*matrix[x][y-1]
                                        - matrix[x+1][y-1]))

                except IndexError:
                    deltaMat[x][y] = 0
    elif method == 'prewitt_diagonal':
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                try:
                    if(x-1 < 0 or y-1 < 0):
                        raise IndexError('Negative value')
                    deltaMat[x][y] = (abs(matrix[x-1][y]
                                        + matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        - matrix[x][y-1]
                                        - matrix[x+1][y-1]
                                        - matrix[x+1][y])
                                        + abs(matrix[x][y+1]
                                        + matrix[x+1][y+1]
                                        + matrix[x+1][y]
                                        - matrix[x][y-1]
                                        - matrix[x-1][y-1]
                                        - matrix[x-1][y]))

                except IndexError:
                    deltaMat[x][y] = 0
    elif method == 'sobel_diagonal':
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                try:
                    if(x-1 < 0 or y-1 < 0):
                        raise IndexError('Negative value')
                    deltaMat[x][y] = (abs(matrix[x-1][y]
                                        + 2*matrix[x-1][y+1]
                                        + matrix[x][y+1]
                                        - matrix[x][y-1]
                                        - 2*matrix[x+1][y-1]
                                        - matrix[x+1][y])
                                        + abs(matrix[x][y+1]
                                        + 2*matrix[x+1][y+1]
                                        + matrix[x+1][y]
                                        - matrix[x][y-1]
                                        - 2*matrix[x-1][y-1]
                                        - matrix[x-1][y]))

                except IndexError:
                    deltaMat[x][y] = 0
    elif method == 'mh':
        sigma = 4
        deltaMat = filters.gaussian_filter(matrix, sigma)
        #print(deltaMat.min(), deltaMat.max())
        #print(deltaMat)
        #return deltaMat
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




parser = argparse.ArgumentParser(description='Image segmentation')

parser.add_argument('image')
parser.add_argument('--roberts', action='store_true')
parser.add_argument('--prewitt', action='store_true')
parser.add_argument('--sobel', action='store_true')
parser.add_argument('--prewitt_diagonal', action='store_true')
parser.add_argument('--sobel_diagonal', action='store_true')
parser.add_argument('--mh', action='store_true')

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

        newMat = detection(matrix, method)

        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



