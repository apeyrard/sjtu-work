#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import argparse
import scipy.signal as ss

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data, dtype=float).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

def getHist(matrix):
    hist = np.zeros(256)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[0]):
            hist[matrix[x][y]] += 1
    return hist

def normHist(hist):
    total = sum(hist)
    for x in range(len(hist)):
        hist[x] = hist[x]/total
    return hist

def otsu(matrix):
    hist = getHist(matrix)
    hist = normHist(hist)
    cumSum = np.zeros(256)
    cumMean = np.zeros(256)
    sigma = np.zeros(256)
    for k in range(256):
        if k == 0:
            cumSum[k] = hist[k]
            cumMean[k] = 0
        else:
            cumSum[k] = cumSum[k-1] + hist[k]
            cumMean[k] = cumMean[k-1] +k*hist[k]

    for k in range(256):
        if cumSum[k] == 0 or cumSum[k] == 1:
            sigma[k] = 0
        else:
            sigma[k] = (cumMean[len(hist)-1]*cumSum[k]- cumMean[k])**2/(cumSum[k] * (1-cumSum[k]))

    sigmaMax = sigma.max()
    nb = 0
    kstar = 0
    for k in range(256):
        if sigma[k] == sigmaMax:
            nb += 1
            kstar += k
    kstar = kstar/nb
    sigmaG = 0
    for k in range(256):
        sigmaG += ((k-cumMean[255])**2)*hist[k]
    #eta = sigma[kstar]/sigmaG

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x][y] < kstar:
                matrix[x][y] = 0
            else:
                matrix[x][y] = 255
    #eta ?
    return matrix

def upperleftmost(matrix):
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x][y] == 255:
                return (x, y)
    return None

def boundary(matrix):
    mask = np.ones((9,9))/(9*9)
    matrix = ss.convolve(matrix, mask, mode="same")
    matrix = otsu(matrix)
    b0 = upperleftmost(matrix)
    c0 = (b0[0], b0[1]-1)

    sequence = []
    sequence.append(b0)

    if matrix[b0[0]-1, b0[1]-1] == 255:
        b1 = (b0[0]-1, b0[1]-1)
        c1 = c0
    elif matrix[b0[0]-1, b0[1]] == 255:
        b1 = (b0[0]-1, b0[1])
        c1 = (b0[0]-1, b0[1]-1)
    elif matrix[b0[0]-1, b0[1]+1] == 255:
        b1 = (b0[0]-1, b0[1]+1)
        c1 = (b0[0]-1, b0[1])
    elif matrix[b0[0], b0[1]+1] == 255:
        b1 = (b0[0], b0[1]+1)
        c1 = (b0[0]-1, b0[1]+1)
    elif matrix[b0[0]+1, b0[1]+1] == 255:
        b1 = (b0[0]+1, b0[1]+1)
        c1 = (b0[0], b0[1]+1)
    elif matrix[b0[0]+1, b0[1]] == 255:
        b1 = (b0[0]+1, b0[1])
        c1 = (b0[0]+1, b0[1]+1)
    elif matrix[b0[0]+1, b0[1]-1] == 255:
        b1 = (b0[0]+1, b0[1]-1)
        c1 = (b0[0]+1, b0[1])

    b = b1
    sequence.append(b1)
    c = c1

    flag = True
    while(flag):
        t = (b[0]-c[0], b[1]-c[1])
        if t == (0, 1):
            n1 = (b[0], b[1]-1)
            n2 = (b[0]-1, b[1]-1)
            n3 = (b[0]-1, b[1])
            n4 = (b[0]-1, b[1]+1)
            n5 = (b[0], b[1]+1)
            n6 = (b[0]+1, b[1]+1)
            n7 = (b[0]+1, b[1])
            n8 = (b[0]+1, b[1]-1)
        elif t == (1, 1):
            n8 = (b[0], b[1]-1)
            n1 = (b[0]-1, b[1]-1)
            n2 = (b[0]-1, b[1])
            n3 = (b[0]-1, b[1]+1)
            n4 = (b[0], b[1]+1)
            n5 = (b[0]+1, b[1]+1)
            n6 = (b[0]+1, b[1])
            n7 = (b[0]+1, b[1]-1)
        elif t == (1, 0):
            n7 = (b[0], b[1]-1)
            n8 = (b[0]-1, b[1]-1)
            n1 = (b[0]-1, b[1])
            n2 = (b[0]-1, b[1]+1)
            n3 = (b[0], b[1]+1)
            n4 = (b[0]+1, b[1]+1)
            n5 = (b[0]+1, b[1])
            n6 = (b[0]+1, b[1]-1)
        elif t == (1, -1):
            n6 = (b[0], b[1]-1)
            n7 = (b[0]-1, b[1]-1)
            n8 = (b[0]-1, b[1])
            n1 = (b[0]-1, b[1]+1)
            n2 = (b[0], b[1]+1)
            n3 = (b[0]+1, b[1]+1)
            n4 = (b[0]+1, b[1])
            n5 = (b[0]+1, b[1]-1)
        elif t == (0, -1):
            n5 = (b[0], b[1]-1)
            n6 = (b[0]-1, b[1]-1)
            n7 = (b[0]-1, b[1])
            n8 = (b[0]-1, b[1]+1)
            n1 = (b[0], b[1]+1)
            n2 = (b[0]+1, b[1]+1)
            n3 = (b[0]+1, b[1])
            n4 = (b[0]+1, b[1]-1)
        elif t == (-1, -1):
            n4 = (b[0], b[1]-1)
            n5 = (b[0]-1, b[1]-1)
            n6 = (b[0]-1, b[1])
            n7 = (b[0]-1, b[1]+1)
            n8 = (b[0], b[1]+1)
            n1 = (b[0]+1, b[1]+1)
            n2 = (b[0]+1, b[1])
            n3 = (b[0]+1, b[1]-1)
        elif t == (-1, 0):
            n3 = (b[0], b[1]-1)
            n4 = (b[0]-1, b[1]-1)
            n5 = (b[0]-1, b[1])
            n6 = (b[0]-1, b[1]+1)
            n7 = (b[0], b[1]+1)
            n8 = (b[0]+1, b[1]+1)
            n1 = (b[0]+1, b[1])
            n2 = (b[0]+1, b[1]-1)
        elif t == (-1, 1):
            n2 = (b[0], b[1]-1)
            n3 = (b[0]-1, b[1]-1)
            n4 = (b[0]-1, b[1])
            n5 = (b[0]-1, b[1]+1)
            n6 = (b[0], b[1]+1)
            n7 = (b[0]+1, b[1]+1)
            n8 = (b[0]+1, b[1])
            n1 = (b[0]+1, b[1]-1)

        if matrix[n1] == 255:
            new = n1
            c = n8
        elif matrix[n2] == 255:
            new = n2
            c = n1
        elif matrix[n3] == 255:
            new = n3
            c = n2
        elif matrix[n4] == 255:
            new = n4
            c = n3
        elif matrix[n5] == 255:
            new = n5
            c = n4
        elif matrix[n6] == 255:
            new = n6
            c = n5
        elif matrix[n7] == 255:
            new = n7
            c = n6
        elif matrix[n8] == 255:
            new = n8
            c = n7

        if b == b0 and new == b1:
            flag = False
        b = new
        sequence.append(b)

    boundary = np.zeros(matrix.shape)
    for item in sequence:
        boundary[item] = 255

    return boundary

parser = argparse.ArgumentParser(description='Boundary following')

parser.add_argument('image')
parser.add_argument('--boundary', action='store_true')

args = parser.parse_args()

try:
    with Image.open(args.image) as im:
        matrix = getMatrix(im)

        if args.boundary:
            newMat = boundary(matrix)


        newIm = Image.new(im.mode, im.size)
        newIm.putdata(getData(newMat))
        newIm.show()

except FileNotFoundError as e:
    sys.exit("Error : file not found")



