#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
#import scipy as sp

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def slice(matrix):
    #slice matrix in 8*8 submatrices
    result = []
    for j in range(int(matrix.shape[0]/8)):
        tmp = []
        for k in range(int(matrix.shape[0]/8)):
            submatrix = np.zeros((8,8))
            for x in range(8):
                for y in range(8):
                    submatrix[x][y] = matrix[8*j + x][8*k + y]
            tmp.append(submatrix)
        result.append(tmp)
    return result

try:
    with Image.open('./lenna.tif') as im:
        matrix = getMatrix(im)
        print(slice(matrix))
except FileNotFoundError as e:
    sys.exit("Error : file not found")



