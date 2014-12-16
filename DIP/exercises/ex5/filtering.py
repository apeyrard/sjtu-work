#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
import cmath
from PIL import Image
import numpy as np

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def preprocessing(matrix):
    newMat = matrix.copy()
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            newMat[x][y] = newMat[x][y]*(-1)**(x+y)
    return newMat

def postprocessing(matrix):
    return preprocessing(matrix)

def rescale(matrix):
    newMat = matrix.copy()
    tmpMin = matrix.min()
    for y in range(matrix.shape[1]):
        for x in range(matrix.shape[0]):
            newMat[x][y] = matrix[x][y]-tmpMin

    tmpMax = newMat.max()

    for y in range(matrix.shape[1]):
        for x in range(matrix.shape[0]):
            newMat[x][y] = newMat[x][y]*255/tmpMax
    return newMat

def blur(matrix, a, b, T):
    blurMat = matrix.copy()
    hor = int(blurMat.shape[1]/2)
    vert = int(blurMat.shape[0]/2)
    for y in range(-hor, hor):
        for x in range(-vert, vert):
            u = x
            v = y
            try:
                blurMat[x][y] = (T/(cmath.pi*(u*a+v*b)))*cmath.sin(cmath.pi*(u*a+v*b))*cmath.exp(-1j*cmath.pi*(u*a+v*b))
            except ZeroDivisionError:
                blurMat[x][y] = 1
    return blurMat

def inverse(matrix):
    newMat = matrix.copy()
    prepMat = preprocessing(newMat)
    fourierMat = np.fft.fft2(prepMat)


    blurredFMat = fourierMat / blur(fourierMat, a=0.1, b=0.1, T=1)

    blurredMat = np.fft.ifft2(blurredFMat)

    blurredPost = postprocessing(blurredMat)

    real = getMatrix(im)
    for y in range(real.shape[1]):
        for x in range(real.shape[0]):
            print(blurredPost[x][y].real)
            real[x][y] = blurredPost[x][y].real

    rescaled = rescale(real)
    return rescaled

def filter(matrix, function):
    result = function(matrix)
    return result

parser = argparse.ArgumentParser(description='Filters images.')

parser.add_argument('image')
parser.add_argument('filter')

args =  parser.parse_args()

if args.filter == 'inverse':
    function = inverse

try:
    im = Image.open(args.image)
except FileNotFoundError as e:
    sys.exit("Error : file not found")

matrix = getMatrix(im)

matrix = filter(matrix, function)

newim = Image.new(im.mode, im.size)
newim.putdata(getData(matrix))
newim.show()

