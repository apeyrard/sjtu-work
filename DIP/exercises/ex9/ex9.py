#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import argparse

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
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
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if method == 'roberts':
                try:
                    deltaMat[x][y] = abs(matrix[x+1][y+1] - matrix[x][y]) + abs(matrix[x+1][y] - matrix[x][y+1])
                except IndexError:
                    deltaMat[x][y] = 0
            elif method == 'prewitt':
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
    return deltaMat




parser = argparse.ArgumentParser(description='Image segmentation')

parser.add_argument('image')
parser.add_argument('--roberts', action='store_true')
parser.add_argument('--prewitt', action='store_true')
parser.add_argument('--sobel', action='store_true')

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

        newMat = detection(matrix, method)

        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



