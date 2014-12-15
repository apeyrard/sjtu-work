#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import math

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def rotate(matrix, theta):
    height, width = matrix.shape
    newMat = matrix.copy()
    center = ((height-1)/2, (width-1)/2)

    for x in range(newMat.shape[0]):
        for y in range(newMat.shape[1]):
            j = x-center[0]
            k = y-center[1]
            l = j*math.cos(theta) + k*math.sin(theta)
            m = j*math.sin(theta)*(-1) + k*math.cos(theta)
            a = l+center[0]
            b = m+center[1]

            #nearest
            a = round(a)
            b = round(b)
            if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                newMat[x][y] = 0
            else:
                newMat[x][y] = matrix[a][b]

    return newMat

def translate(matrix, delta):
    newMat = matrix.copy()

    for x in range(newMat.shape[0]):
        for y in range(newMat.shape[1]):
            a, b = x-delta[0], y-delta[1]
            if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                newMat[x][y] = 0
            else:
                newMat[x][y] = matrix[a][b]
    return newMat

def rescale(matrix, scale):
    height, width = matrix.shape
    newMat = matrix.copy()
    center = ((height-1)/2, (width-1)/2)

    for x in range(newMat.shape[0]):
        for y in range(newMat.shape[1]):
            j = x-center[0]
            k = y-center[1]
            l = j/scale
            m = k/scale
            a = l+center[0]
            b = m+center[1]

            #nearest
            a = round(a)
            b = round(b)
            if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                newMat[x][y] = 0
            else:
                newMat[x][y] = matrix[a][b]

    return newMat



try:
    im = Image.open("./ray_trace_bottle.tif")
except FileNotFoundError as e:
    sys.exit("Error : file not found")

matrix = getMatrix(im)

transform = rescale(matrix, 2)

newMat = Image.new(im.mode, im.size)

newMat.putdata(getData(transform))

newMat.show()


