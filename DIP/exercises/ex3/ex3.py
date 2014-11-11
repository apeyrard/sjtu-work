#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import math

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(width,height)
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

def ideal(matrix, cutoff, function):
    newMat = matrix.copy()
    center = (math.floor(newMat.shape[0]/2), math.floor(newMat.shape[1]/2))
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            dist = math.sqrt((x-center[0])**2+(y-center[1])**2)
            if function == 'low':
                if dist > cutoff:
                    newMat[x][y] = 0+0j
            if function == 'high':
                if dist < cutoff:
                    newMat[x][y] = 0+0j
    return newMat

def butter(matrix, order, cutoff, function):
    newMat = matrix.copy()
    center = (math.floor(newMat.shape[0]/2), math.floor(newMat.shape[1]/2))
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            dist = math.sqrt((x-center[0])**2+(y-center[1])**2)
            if function == 'low':
                newMat[x][y] = newMat[x][y] * (1/(1+(dist/cutoff)**(2*order)))

            if function == 'high':
                newMat[x][y] = newMat[x][y] * (1-(1/(1+(dist/cutoff)**(2*order))))
    return newMat

def gauss(matrix, cutoff, function):
    newMat = matrix.copy()
    center = (math.floor(newMat.shape[0]/2), math.floor(newMat.shape[1]/2))
    for y in range(newMat.shape[1]):
        for x in range(newMat.shape[0]):
            dist = math.sqrt((x-center[0])**2+(y-center[1])**2)
            if function == 'low':
                newMat[x][y] = newMat[x][y] * (math.exp(-(dist**2)/(2*(cutoff**2))))
            if function == 'high':
                newMat[x][y] = newMat[x][y] * (1- (math.exp(-(dist**2)/(2*(cutoff**2)))))
    return newMat

try:
    im = Image.open("./characters_test_pattern.tif")
except FileNotFoundError as e:
    sys.exit("Error : file not found")

imIdealLow = Image.new(im.mode, im.size)
imIdealHigh = Image.new(im.mode, im.size)
imButterLow = Image.new(im.mode, im.size)
imButterHigh = Image.new(im.mode, im.size)
imGaussLow = Image.new(im.mode, im.size)
imGaussHigh = Image.new(im.mode, im.size)

matrix = getMatrix(im)

prepMat = preprocessing(matrix)
fourierMat = np.fft.fft2(prepMat)

idealLowF = ideal(fourierMat, 30, 'low')
idealHighF = ideal(fourierMat, 30, 'high')
butterLowF = butter(fourierMat, 2, 30, 'low')
butterHighF = butter(fourierMat, 2, 30, 'high')
gaussLowF = gauss(fourierMat, 5, 'low')
gaussHighF = gauss(fourierMat, 5, 'high')

idealLow = np.fft.ifft2(idealLowF)
idealHigh = np.fft.ifft2(idealHighF)
butterLow = np.fft.ifft2(butterLowF)
butterHigh = np.fft.ifft2(butterHighF)
gaussLow = np.fft.ifft2(gaussLowF)
gaussHigh = np.fft.ifft2(gaussHighF)

postIdealLow = postprocessing(idealLow)
postIdealHigh = postprocessing(idealHigh)
postButterLow = postprocessing(butterLow)
postButterHigh = postprocessing(butterHigh)
postGaussLow = postprocessing(gaussLow)
postGaussHigh = postprocessing(gaussHigh)

imIdealLow.putdata(getData(postIdealLow))
imIdealHigh.putdata(getData(postIdealHigh))
imButterLow.putdata(getData(postButterLow))
imButterHigh.putdata(getData(postButterHigh))
imGaussLow.putdata(getData(postGaussLow))
imGaussHigh.putdata(getData(postGaussHigh))

#imIdealLow.show()
#imIdealHigh.show()
#imButterLow.show()
#imButterHigh.show()
imGaussLow.show()
imGaussHigh.show()


