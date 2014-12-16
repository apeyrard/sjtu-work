#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
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

def inverse(matrix):
    return matrix

def filter(matrix, function):
    if function == 'inverse':
        result = inverse(matrix)
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

