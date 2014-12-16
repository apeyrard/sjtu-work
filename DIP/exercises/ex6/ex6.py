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

def bilinear(a, b, matrix):
    left = math.floor(b)
    right = math.ceil(b)
    down = math.ceil(a)
    up = math.floor(a)

    #horizontal linear interpolation
    #up
    try:
        if up < 0 or right < 0 or left < 0:
            raise IndexError("Negative index")
        if right == left:
            upMidValue = matrix[up][right]
        else:
            upMidValue = (((right-b)/(right-left))*matrix[up][right] + ((b-left)/(right-left))*matrix[up][left])
    except IndexError:
        upMidValue = None

    #down
    try:
        if down < 0 or right < 0 or left < 0:
            raise IndexError("Negative index")
        if right == left:
            downMidValue = matrix[down][right]
        else:
            downMidValue = (((right-b)/(right-left))*matrix[down][right] + ((b-left)/(right-left))*matrix[down][left])
    except IndexError:
        downMidValue = None

    if upMidValue is None and downMidValue is None:
        final = 0
    elif upMidValue is None:
        final = downMidValue
    elif downMidValue is None:
        final = upMidValue
    elif upMidValue == downMidValue:
        final = upMidValue
    else:
        final = (((down-a)/(down-up))*downMidValue + ((a-up)/(down-up))*upMidValue)

    return final

def translate(matrix, delta, method):
    newMat = matrix.copy()

    for x in range(newMat.shape[0]):
        for y in range(newMat.shape[1]):
            a, b = x-delta[0], y-delta[1]
            if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                newMat[x][y] = 0
            else:
                if method == 'nearest':
                    newMat[x][y] = matrix[round(a)][round(b)]
                else:
                    newMat[x][y] = bilinear(a, b, matrix)

    return newMat

def rotate(matrix, theta, method):
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

            if method == 'nearest':
                a = round(a)
                b = round(b)
                if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                    newMat[x][y] = 0
                else:
                    newMat[x][y] = matrix[a][b]
            else:
                final = bilinear(a, b, matrix)
                newMat[x][y] = final
    return newMat

def rescale(matrix, scale, method):
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

            if method == 'nearest':
                a = round(a)
                b = round(b)
                if a<0 or b<0 or a>=newMat.shape[0] or b>=newMat.shape[1]:
                    newMat[x][y] = 0
                else:
                    newMat[x][y] = matrix[a][b]
            else:
                final = bilinear(a, b, matrix)
                newMat[x][y] = final
    return newMat

try:
    im = Image.open("./ray_trace_bottle.tif")
except FileNotFoundError as e:
    sys.exit("Error : file not found")

matrix = getMatrix(im)

#method = 'nearest'
method = 'bilinear'
transform = rotate(matrix, math.pi/9, method)

newMat = Image.new(im.mode, im.size)

newMat.putdata(getData(transform))

newMat.show()


