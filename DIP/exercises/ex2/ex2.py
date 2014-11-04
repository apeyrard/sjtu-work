#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys

from PIL import Image
import numpy as np
import math

try:
    im = Image.open("./skeleton_orig.tif")
except FileNotFoundError as e:
    sys.exit("Error : file not found")

def applyFilter(filterMatrix, data, width, height):
    print("Applying filter : " + str(filterMatrix))
    newData = list()
    for y in range(height):
        for x in range(width):
            newPix = 0
            for k in range(filterMatrix.shape[0]):
                for j in range(filterMatrix.shape[1]):
                    a=x+k-math.floor(filterMatrix.shape[0]/2)
                    b=y+j-math.floor(filterMatrix.shape[1]/2)
                    try:
                        tmpVal=data[a+b*width]
                    except IndexError:
                        tmpVal=0
                    newPix += tmpVal*filterMatrix[k,j]
            newData.append(newPix)

    return newData

def stretch(data):
    newData = []
    tmpData = []

    minPix = min(data)
    for px in data:
        tmpData.append(px - minPix)

    maxPix = max(tmpData)
    for px in tmpData:
        newData.append(px*255/maxPix)
    return newData


lFilter = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1]).reshape(3,3)

width, height = im.size
data = applyFilter(lFilter, im.getdata(), width, height)
im2 = Image.new(im.mode, im.size)
im2.putdata(data)

datast = stretch(data)
im2st = Image.new(im.mode, im.size)
im2st.putdata(datast)

#rescaled laplacian + original
data4 = list(im.getdata())
data3 = stretch([(data4[i] + datast[i]) for i in range(len(data4))])
im3 = Image.new(im.mode, im.size)
im3.putdata(data3)

#sobel gradient
sobelFilterX = np.array([-1, 0, +1, -2, 0, +2, -1, 0, +1]).reshape(3,3)
sobelFilterY = np.array([+1, +2, +1, 0, 0, 0, -1, -2, -1]).reshape(3,3)
datax = applyFilter(sobelFilterX, im.getdata(), width, height)
datay = applyFilter(sobelFilterY, im.getdata(), width, height)
dataSobel = [round(math.sqrt(datax[i]**2 + datay[i]**2)) for i in range(len(datax))]
imSobel = Image.new(im.mode, im.size)
imSobel.putdata(dataSobel)

#smoothing sobel
smoothFilter = np.array([1/25, 1/25, 1/25, 1/25, 1/25,
                        1/25, 1/25, 1/25, 1/25, 1/25,
                        1/25, 1/25, 1/25, 1/25, 1/25,
                        1/25, 1/25, 1/25, 1/25, 1/25,
                        1/25, 1/25, 1/25, 1/25, 1/25]).reshape(5,5)

smoothSobel = applyFilter(smoothFilter, dataSobel, width, height)
imSmoothSobel = Image.new(im.mode, im.size)
imSmoothSobel.putdata(smoothSobel)

#productof smooth and sum
dataMask = stretch([round(data3[i]*smoothSobel[i]) for i in range(len(data3))])
imMask = Image.new(im.mode, im.size)
imMask.putdata(dataMask)

#sum of original and mask
data=im.getdata()
dataSharp = [round(data[i]+dataMask[i]) for i in range(len(data))]
imSharp = Image.new(im.mode, im.size)
imSharp.putdata(dataSharp)

#power law
gamma = 0.5
const = 1
dataPower = stretch([round(1*dataSharp[i]**gamma) for i in range(len(dataSharp))])
imPower = Image.new(im.mode, im.size)
imPower.putdata(dataPower)

im.show()
im2st.show()
im3.show()
imSobel.show()
imSmoothSobel.show()
imMask.show()
imSharp.show()
imPower.show()
