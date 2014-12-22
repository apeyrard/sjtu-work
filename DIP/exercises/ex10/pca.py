#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
import os
from PIL import Image
import numpy as np

size = None
matrix_x = None
for image in os.listdir('./washington'):
    try:
        with Image.open(os.path.join('./washington',image)) as im:
            imgVector = np.array(list(im.getdata()))
            imgVector = imgVector.reshape(1, imgVector.shape[0])
            try:
                matrix_x = np.vstack((matrix_x, imgVector))
            except:
                matrix_x = imgVector
    except FileNotFoundError as e:
        sys.exit("Error : file not found")

#matrix_x = np.array([[0,1,1,1],
                        #[0,0,1,0],
                        #[0,0,0,1]
                        #])

#mean vector
K = matrix_x.shape[1]
print('K', K)
nb = matrix_x.shape[0]
print('nb', nb)
mx = np.zeros((nb, 1))
for x in range(K):
    for y in range(nb):
        mx[y] += matrix_x[y, x]
mx = mx/K

#covar matrix
cx = np.zeros((nb,nb))
for x in range(K):
    tmp = (matrix_x[:,x])
    tmp = tmp.reshape(tmp.shape[0],1)
    cx += np.dot(tmp,tmp.T) - np.dot(mx,mx.T)
cx = cx/K

eigenvalues, eigenvectors = np.linalg.eig(cx)
#tri
eival = np.zeros(eigenvalues.shape)
eivec = np.zeros(eigenvectors.shape)
j = 0
for _ in range(nb):
    maxval = eigenvalues.max()
    for i in range(eigenvalues.shape[0]):
        val = eigenvalues[i]
        if val == maxval:
            eival[j] = val
            eigenvalues[i] = 0
            eivec[j] = eigenvectors[i]
            j += 1
            break

#pruning eivec
pruning = 2
eivec = eivec[:pruning,:]
print(eivec)

matrix_y = np.zeros((pruning, matrix_x.shape[1]))
for i in range(K):
    tmp = (matrix_x[:,i]).reshape(nb, 1)
    truc = np.dot(eivec,(tmp-mx))
    matrix_y[:, i] = truc.reshape(truc.shape[0])


#reconstruction
matrix_x2 = np.zeros(matrix_x.shape)
for i in range(K):
    tmp = (matrix_y[:,i])
    tmp = tmp.reshape(tmp.shape[0], 1)
    matrix_x2[:, i] = np.array((np.dot(eivec.T,tmp)+mx).reshape(nb))

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

data = np.vsplit(matrix_x2, 6)
for i,item in enumerate(data):
    item = list(item.reshape(item.shape[1]))
    newIm = Image.new(im.mode, im.size)
    newIm.putdata(item)
    newIm.show()

    diff = item - matrix_x[i]
    epsilon = 0.1
    print(diff)
    for j,val in enumerate(diff):
        if abs(val) < epsilon:
            diff[j] = 0
    print(diff)
    diff = rescale(diff)
    newIm = Image.new(im.mode, im.size)
    newIm.putdata(list(diff))
    newIm.show()





