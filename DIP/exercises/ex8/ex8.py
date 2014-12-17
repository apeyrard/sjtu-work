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

def toBinary(matrix):
    return matrix/255

def hasNeighbour(matrix, x, y, value):
    for i in range(-1,2):
        for j in range(-1,2):
            try:
                if x+i < 0 or y+j < 0:
                    raise IndexError("Negative value")
                if matrix[x+i][y+j] == value:
                    return True
            except IndexError:
                pass
    return False

def dilate(matrix, mask):
    matrix = toBinary(matrix)
    toAdd = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x][y] == 1:
                if hasNeighbour(matrix, x, y, 0):
                    for j in range(mask.shape[0]):
                        for k in range(mask.shape[1]):
                            try:
                                if (x+j-1)<0 or (y+k-1)<0:
                                    raise IndexError("Negative value")
                                if matrix[x+j-1][y+k-1] == 1 or mask[j][k] == 1:
                                    toAdd[x+j-1][y+k-1] = 1
                            except IndexError:
                                pass
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            matrix[x][y] = max(matrix[x][y], toAdd[x][y])
    return rescale(matrix)

def erode(matrix, mask):
    matrix = toBinary(matrix)
    toDel = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x][y] == 0:
                if hasNeighbour(matrix, x, y, 1):
                    for j in range(mask.shape[0]):
                        for k in range(mask.shape[1]):
                            try:
                                if (x+j-1)<0 or (y+k-1)<0:
                                    raise IndexError("Negative value")
                                if mask[j][k] == 1:
                                    toDel[x+j-1][y+k-1] = 1
                            except IndexError:
                                pass
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            matrix[x][y] = max(matrix[x][y] - toDel[x][y], 0)
    return rescale(matrix)

def open(matrix, mask):
    return dilate(erode(matrix,mask),mask)

def close(matrix, mask):
    return erode(dilate(matrix,mask),mask)

parser = argparse.ArgumentParser(description='Morphological image processing')

parser.add_argument('image')
parser.add_argument('--dilate', action='store_true')
parser.add_argument('--erode', action='store_true')

parser.add_argument('--open', action='store_true')
parser.add_argument('--close', action='store_true')

args = parser.parse_args()

try:
    with Image.open(args.image) as im:
        matrix = getMatrix(im)

        mask = np.ones((3,3))
        if args.dilate:
            newMat = dilate(matrix, mask)
        if args.erode:
            newMat = erode(matrix, mask)
        if args.open:
            newMat = open(matrix, mask)
        if args.close:
            newMat = close(matrix, mask)

        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



