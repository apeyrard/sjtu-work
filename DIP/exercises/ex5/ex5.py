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

def inverse_filter(filter, threshold):
    return np.vectorize(lambda x: 1./x if np.abs(x) >= threshold else 1.0, otypes=[np.complex])(filter)

def weiner(matrix, H, N):
    Sxx = np.abs(matrix) ** 2

    if N is not None:
        Snn = N ** 2
    else:
        Snn = None

    if Snn is not None:
        m, n = H.shape

        d=0.003

        result = np.empty((m,n), dtype=complex)

        conjHT = np.conj(H).T
        denoms = (np.abs(H)**2+Snn/Sxx)

        for u in range(m):
            for v in range(n):
                denom = denoms[u, v]
                result[u, v] = conjHT[u, v] / denom if np.abs(denom)>=d else 1.0
        return result
    else:
        return inverse_filter(H)

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

            deblurMat = inverse_filter(blur(fourierMat, a=0.1, b=0.1, T=1), threshold = 1.e-2)

            blurredFMat = fourierMat * deblurMat

            blurredMat = np.fft.ifft2(blurredFMat)

            blurred = Image.new(im.mode, im.size)

            blurredPost = postprocessing(blurredMat)

            real = blurredPost.real

            rescaled = rescale(real)

            blurred.putdata(getData(rescaled))

            blurred.save(args.output)
except FileNotFoundError as e:
    sys.exit("Error : file not found")



