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
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            matrix[x][y] = 1 if matrix[x][y] > 203 else 0
    return matrix

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
    return matrix

def erode(matrix, mask):
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
    return matrix

def open(matrix, mask):
    return dilate(erode(matrix,mask),mask)

def close(matrix, mask):
    return erode(dilate(matrix,mask),mask)

def boundary(matrix, mask):
    original = matrix.copy()
    eroded = erode(matrix, mask)
    diff = original - eroded
    return diff

def union(mat1, mat2):
    for x in range(mat1.shape[0]):
        for y in range(mat1.shape[1]):
            mat1[x][y] = max(mat1[x][y],mat2[x][y])
    return mat1

def intersection(mat1, mat2):
    for x in range(mat1.shape[0]):
        for y in range(mat1.shape[1]):
            mat1[x][y] = min(mat1[x][y],mat2[x][y])
    return mat1

def filling(matrix, x, y):

    mask = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])

    negative = np.ones(matrix.shape) - matrix

    start = np.zeros(matrix.shape)
    start[x][y] = 1

    current = start

    iter = 1
    while(True):
        print("iteration : ", iter)
        iter += 1
        #if iter == 20:
            #return union(current,matrix)
        old = current.copy()
        current = dilate(current, mask)
        current = intersection(current, negative)
        if np.array_equal(old,current):
            return union(current,matrix)

def extraction(matrix, x, y):

    mask = np.ones((3,3))
    start = np.zeros(matrix.shape)
    start[x][y] = 1

    current = start

    iter = 1
    while(True):
        print("iteration : ", iter)
        iter += 1
        #if iter == 100:
            #return current
        old = current.copy()
        current = dilate(current, mask)
        current = intersection(current, matrix)
        if np.array_equal(old,current):
            return current

parser = argparse.ArgumentParser(description='Morphological image processing')

parser.add_argument('image')
parser.add_argument('--dilate', action='store_true')
parser.add_argument('--erode', action='store_true')

parser.add_argument('--open', action='store_true')
parser.add_argument('--close', action='store_true')

parser.add_argument('--boundary', action='store_true')
parser.add_argument('--filling', action='store_true')
parser.add_argument('--extraction', action='store_true')

args = parser.parse_args()

try:
    with Image.open(args.image) as im:
        matrix = getMatrix(im)
        matrix = toBinary(matrix)

        mask = np.ones((3,3))
        if args.dilate:
            newMat = dilate(matrix, mask)
        if args.erode:
            newMat = erode(matrix, mask)
        if args.open:
            newMat = open(matrix, mask)
        if args.close:
            newMat = close(matrix, mask)
        if args.boundary:
            newMat = boundary(matrix, mask)
        if args.filling:
            newMat = filling(matrix, 50, 50)
        if args.extraction:
            newMat = extraction(matrix, 160, 356)

        newMat = rescale(newMat)

        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



