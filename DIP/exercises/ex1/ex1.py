#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Choose how to process input image.')
parser.add_argument('image')

args =  parser.parse_args()

def getHist(data, maxPix):
    hist = np.zeros(maxPix, dtype=int)
    for x in data:
            hist[x] += 1
    return hist

try:
    with Image.open(args.image) as im:
        #getting the list of pixel values
        data = list(im.getdata())

        maxPix = 256
        #getting histogram
        histList = getHist(data, maxPix)

        #plotting histogram
        plt.bar(np.arange(maxPix), histList)
        plt.ylabel('Nb of pixels')
        plt.xlabel('Value')
        plt.xlim(0, maxPix)
        plt.show()

        #total number of pixels
        total = sum(histList)

        #maps value to new value in [0, 255] range
        def transform(value):
            tmpSum = 0
            for j in range(value):
                tmpSum += histList[j]/total
            return tmpSum*maxPix

        #getting cumulative distribution function
        cdf = np.zeros(maxPix)
        for i,val in enumerate(histList):
            if i != 0:
                cdf[i] = cdf[i-1] + val

        #plotting cdf
        plt.plot(cdf)
        plt.ylabel('Value of new pixel')
        plt.xlabel('Value of initial pixel')
        plt.xlim(0, maxPix)
        plt.show()

        #new image
        newim = Image.new(im.mode, im.size)
        ptr = newim.load()
        for i, pixel in enumerate(data):
            newValue = round(transform(pixel))
            ptr[i%im.size[0], i//im.size[0]] = newValue
        newim.show()

        data = list(newim.getdata())

        histList = getHist(data, maxPix)

        plt.bar(np.arange(maxPix), histList)
        plt.ylabel('Nb of pixels')
        plt.xlabel('Value')
        plt.xlim(0, maxPix)

        plt.show()
except FileNotFoundError as e:
    sys.exit("Error : file not found")
