#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import cmath


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

def blur(matrix, a, b, T):
    newMat = matrix.copy()
    blurMat = matrix.copy()
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            u = x+1
            v = y+1
            blurMat[x][y] = (T/(cmath.pi*(u*a+v*b)))*cmath.sin(cmath.pi*(u*a+v*b))*cmath.exp(-1j*cmath.pi*(u*a+v*b))
    #return np.dot(blurMat,newMat)
    return blurMat*newMat

try:
    im = Image.open("./book_cover.jpg")
except FileNotFoundError as e:
    sys.exit("Error : file not found")

matrix = getMatrix(im)
prepMat = matrix#preprocessing(matrix)
fourierMat = np.fft.fft2(prepMat)

blurredFMat = blur(fourierMat, a=0.1, b=0.1, T=1)

blurredMat = np.fft.ifft2(blurredFMat)


blurred = Image.new(im.mode, im.size)



blurredPost = blurredMat#postprocessing(blurredMat)

blurred.putdata(getData(blurredPost))

blurred.show()


