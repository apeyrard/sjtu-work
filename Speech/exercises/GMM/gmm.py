#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# load training data
trainFile = open('train.txt', 'r')
data = [[],[]]
for line in trainFile:
    tmp=line.split()
    if tmp[2] == '1':
        data[0].append((float(tmp[0]),float(tmp[1])))
    elif tmp[2] == '2':
        data[1].append((float(tmp[0]),float(tmp[1])))

class1 = np.array(data[0])
class2 = np.array(data[1])
total = np.array(data[0] + data[1])
# K-means
K = 4 # arbitrary, obtained by looking at data
D = 2 # 2 dimensions
means = (class1.max() - class1.min()) * np.random.random((K,D)) + class1.min()
assign = np.zeros((len(class1)), dtype=int)

while(True):
    # Expectation step
    oldAssign = assign.copy()
    for j,point in enumerate(class1):
        minDist = None
        cluster = None
        for i,mean in enumerate(means):
            dist = (point-mean)**2
            dist = np.sum(dist)
            dist = np.sqrt(dist)
            if minDist is None or dist < minDist:
                minDist = dist
                cluster = i
        assign[j] = cluster

    if not (oldAssign - assign).any():
        break

    # Maximization step
    for i,mean in enumerate(means):
        num = np.array([0, 0], dtype=float)
        denom = 0
        for j,point in enumerate(class1):
            if assign[j] == i:
                num += point
                denom +=1
        if denom != 0:
            means[i] = num/denom
        else:
            means[i] = (class1.max() - class1.min()) * np.random.random(D) + class1.min()

# Plot K-means result
plt.scatter(class1[:,0], class1[:,1], c=assign)
plt.scatter(means[:,0], means[:,1], c=u'g')
plt.show()

#GMM

#plot data
#plt.scatter(class1[:,0], class1[:,1], c=u'b')
#plt.scatter(class2[:,0], class2[:,1], c=u'r')
#plt.scatter(total[:,0], total[:,1], c=u'g')
#plt.show()
