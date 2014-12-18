#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
import os
import cv2 as cv
import numpy as np

matrix_test = None
for image in os.listdir('./washington'):
    imgraw = cv.imread(os.path.join('./washington', image), 0)
    imgvector = imgraw.reshape(imgraw.shape[0]*imgraw.shape[1])
    try:
        matrix_test = np.vstack((matrix_test, imgvector))
    except:
        matrix_test = imgvector

#PCA
mean, eigenvectors = cv.PCACompute(matrix_test, np.mean(matrix_test, axis=0).reshape(1,-1), maxComponents=2)
