#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import scipy.fftpack as sp
import argparse

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

def slice(matrix):
    #slice matrix in 8*8 submatrices
    result = np.zeros((height/8, width/8), dtype=object)
    for j in range(int(matrix.shape[0]/8)):
        for k in range(int(matrix.shape[1]/8)):
            submatrix = np.zeros((8,8))
            for x in range(8):
                for y in range(8):
                    submatrix[x][y] = matrix[8*j + x][8*k + y]
            result[j][k] = submatrix
    return result

def deslice(matrix):
    result = np.zeros((matrix.shape[0]*8, matrix.shape[1]*8))
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            for x in range(8):
                for y in range(8):
                    result[x +8*j][y+8*k] = matrix[j][k][x][y]
    return result

def applyMask(matrix, mask):
    newMat = matrix.copy()
    height, width = matrix.shape
    for j in range(height):
        for k in range(width):
                newMat[j][k] = newMat[j][k] * mask
    return newMat

parser = argparse.ArgumentParser(description='Compression based on DCT or wavelets')

parser.add_argument('image')
parser.add_argument('--showdiff', action='store_true')
parser.add_argument('--dct', action='store_true')
parser.add_argument('--threshold', action='store_true')

args = parser.parse_args()

try:
    with Image.open(args.image) as im:
        matrix = getMatrix(im)

        if args.dct:
            height, width = matrix.shape
            sliced = slice(matrix)
            for x in range(int(height/8)):
                for y in range(int(width/8)):
                    for j in range(8):
                        sliced[x][y][j,:] = sp.dct(sliced[x][y][j,:],norm='ortho')
                    for j in range(8):
                        sliced[x][y][:,j] = sp.dct(sliced[x][y][:,j],norm='ortho')
            if not args.threshold:
                mask=np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])
            else:
                mask=np.array([[1, 1, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])

            sliced = applyMask(sliced,mask)
            for x in range(int(height/8)):
                for y in range(int(width/8)):
                    for j in range(8):
                        sliced[x][y][:,j] = sp.idct(sliced[x][y][:,j],norm='ortho')
                    for j in range(8):
                        sliced[x][y][j,:] = sp.idct(sliced[x][y][j,:],norm='ortho')
            test =  deslice(sliced)
            newIm = Image.new(im.mode, im.size)
            newIm.putdata(getData(test))
            newIm.show()
            if args.showdiff:
                diff = matrix-test
                newIm = Image.new(im.mode, im.size)
                newIm.putdata(getData(diff))
                newIm.show()




except FileNotFoundError as e:
    sys.exit("Error : file not found")



