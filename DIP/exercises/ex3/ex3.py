#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import math
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
    if order is None:
        print("Order must be specified for butterworth filter")
        sys.exit(1)
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

parser = argparse.ArgumentParser(description='Filtering in frequency domain')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--butterworth', action='store_true')
parser.add_argument('--gaussian', action='store_true')
parser.add_argument('--highpass', action='store_true')
parser.add_argument('--lowpass', action='store_true')
parser.add_argument('cutoff', type=float)
parser.add_argument('--order', type=float)
parser.add_argument('image')

args =  parser.parse_args()

try:
    with Image.open(args.image) as im:
        if args.lowpass:
            filtering = 'low'
        else:
            filtering = 'high'
        imNew = Image.new(im.mode, im.size)

        matrix = getMatrix(im)

        prepMat = preprocessing(matrix)
        fourierMat = np.fft.fft2(prepMat)

        if args.ideal:
            imageF = ideal(fourierMat, args.cutoff, filtering)
        elif args.butterworth:
            imageF = butter(fourierMat, args.order, args.cutoff, filtering)
        else:
            imageF = gauss(fourierMat, args.cutoff, filtering)

        newImage = np.fft.ifft2(imageF)

        postNew = postprocessing(newImage)

        imNew.putdata(getData(postNew))

        imNew.show()
except FileNotFoundError as e:
    sys.exit("Error : file not found")

