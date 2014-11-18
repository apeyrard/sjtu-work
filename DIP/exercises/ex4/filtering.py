#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
import math
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

def identity(matrix, x, y, *args):
    return matrix[x][y]

def arithmeticMean(matrix, x, y, m, n):
    if m%2 ==0 or n%2==0:
        print("must be odd")
    else:
        m = math.floor(m/2)
        n = math.floor(n/2)
        tmp = 0
        counter = 0
        for j in range(-m,m+1):
            for k in range(-n,n+1):
                try:
                    if x+j < 0 or y+k < 0:
                        raise IndexError()
                    tmp += matrix[x+j][y+k]
                    counter += 1
                except IndexError:
                    pass
    return tmp/counter

def geoMean(matrix, x, y, m, n):
    if m%2 ==0 or n%2==0:
        print("must be odd")
    else:
        m = math.floor(m/2)
        n = math.floor(n/2)
        tmp = 1
        counter = 0
        for j in range(-m,m+1):
            for k in range(-n,n+1):
                try:
                    if x+j < 0 or y+k < 0:
                        raise IndexError()
                    tmp *= (matrix[x+j][y+k]/255)
                    counter += 1
                except IndexError:
                    pass
    return tmp**(1/counter)*255

def harmoMean(matrix, x, y, m, n):
    if m%2 ==0 or n%2==0:
        print("must be odd")
    else:
        m = math.floor(m/2)
        n = math.floor(n/2)
        tmp = 0
        counter = 0
        for j in range(-m,m+1):
            for k in range(-n,n+1):
                try:
                    if x+j < 0 or y+k < 0:
                        raise IndexError()
                    tmp += 1/((matrix[x+j][y+k]/255))
                    counter += 1
                except IndexError:
                    pass
    return (counter/tmp)*255

def contraMean(matrix, x, y, m, n, order):
    if m%2 ==0 or n%2==0:
        print("must be odd")
    else:
        m = math.floor(m/2)
        n = math.floor(n/2)
        tmp = 0
        tmp2 = 0
        for j in range(-m,m+1):
            for k in range(-n,n+1):
                try:
                    if x+j < 0 or y+k < 0:
                        raise IndexError()
                    tmp += (matrix[x+j][y+k]/255)**(order+1)
                    tmp2 += (matrix[x+j][y+k]/255)**(order)
                except IndexError:
                    pass
    if tmp2 == 0:
        return 0
    else:
        return (tmp/tmp2)*255

def filter(matrix, function, *args):
    newMat = matrix.copy()
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            newMat[x][y] = function(matrix, x, y, *args)
    return newMat

parser = argparse.ArgumentParser(description='Adds uniform noise to image.')

parser.add_argument('image')
parser.add_argument('filter')

args =  parser.parse_args()

if args.filter == 'identity':
    function = identity
elif args.filter == 'arith':
    function = arithmeticMean
elif args.filter == 'geo':
    function = geoMean
elif args.filter == 'harmo':
    function = harmoMean
elif args.filter == 'contra':
    function = contraMean

try:
    im = Image.open(args.image)
except FileNotFoundError as e:
    sys.exit("Error : file not found")

matrix = getMatrix(im)

matrix = filter(matrix, function, 3, 3, -2)

newim = Image.new(im.mode, im.size)
newim.putdata(getData(matrix))
newim.show()

