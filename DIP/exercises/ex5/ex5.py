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

def preprocessing(matrix):
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            matrix[x][y] = matrix[x][y]*(-1)**(x+y)
    return matrix

def postprocessing(matrix):
    return preprocessing(matrix)

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

def blur(matrix, a, b, T):
    m, n = matrix.shape
    u, v = np.ogrid[-m/2:m/2, -n/2:n/2]
    x = u * a + v * b
    result = T * np.sinc(x) * np.exp(-1j*np.pi*x)
    return result

def filter(matrix, function):
    result = function(matrix)
    return result

parser = argparse.ArgumentParser(description='Blurs or filters images')

parser.add_argument('action', choices=['blur', 'filter'])
parser.add_argument('image')
parser.add_argument('output')

args= parser.parse_args()

try:
    with Image.open(args.image) as im:
        if args.action == 'blur':
            matrix = getMatrix(im)
            prepMat = preprocessing(matrix)
            fourierMat = np.fft.fft2(prepMat)
            blurredFMat = fourierMat * blur(fourierMat, a=0.1, b=0.1, T=1)
            blurredMat = np.fft.ifft2(blurredFMat)

            blurredPost = postprocessing(blurredMat)

            real = blurredPost.real

            rescaled = rescale(real)

            blurred = Image.new(im.mode, im.size)
            blurred.putdata(getData(rescaled))

            blurred.save(args.output)
        if args.action == 'filter':
            matrix = getMatrix(im)
            prepMat = preprocessing(matrix)
            fourierMat = np.fft.fft2(prepMat)

            deblurMat = blur(fourierMat, a=0.1, b=0.1, T=1)

            x,y = deblurMat.shape
            for j in range(x):
                for k in range(y):
                    if abs(deblurMat[j][k]) < 1e-2:
                        deblurMat[j][k] = 1
            blurredFMat = fourierMat / deblurMat

            blurredMat = np.fft.ifft2(blurredFMat)

            blurred = Image.new(im.mode, im.size)

            blurredPost = postprocessing(blurredMat)

            real = blurredPost.real

            rescaled = rescale(real)

            blurred.putdata(getData(rescaled))

            blurred.save(args.output)
except FileNotFoundError as e:
    sys.exit("Error : file not found")



