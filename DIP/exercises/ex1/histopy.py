#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Choose how to process input image.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--hist', action='store_true', default=False, help='Draw image histogram')
group.add_argument('--enhance', action='store_true', default=False, help='Enhance image')

parser.add_argument('image')

args =  parser.parse_args()

try:
    im = Image.open(args.image)
except FileNotFoundError as e:
    sys.exit("Error : file not found")

data = list(im.getdata())

maxPix = 0
for pixel in data:
    if pixel > maxPix:
        maxPix = pixel

histArray = [0] * (maxPix+1)
for pixel in data:
    histArray[pixel] += 1

histList = list(histArray)

plt.bar(np.arange(maxPix+1), histList)
plt.ylabel('Nb of pixels')
plt.xlabel('Value')
plt.xlim(0, maxPix)

plt.show()

if args.enhance==True:
    total = sum(histList)
    def transform(value):
        tmpSum = 0
        for j in range(value):
            tmpSum += histList[j]/total
        return tmpSum

    newim = Image.new(im.mode, im.size)
    ptr = newim.load()
    for i, pixel in enumerate(data):
        newValue = round(transform(pixel)*maxPix)
        ptr[i%im.size[0], i//im.size[0]] = newValue
    newim.show()

    data = list(newim.getdata())

    maxPix = 0
    for pixel in data:
        if pixel > maxPix:
            maxPix = pixel

    histArray = [0] * (maxPix+1)
    for pixel in data:
        histArray[pixel] += 1

    histList = list(histArray)

    plt.bar(np.arange(maxPix+1), histList)
    plt.ylabel('Nb of pixels')
    plt.xlabel('Value')
    plt.xlim(0, maxPix)

    plt.show()
