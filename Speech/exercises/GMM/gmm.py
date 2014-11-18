#!/usr/bin/env python2
# -*- coding: UTF-8 -*-


# load training data
trainFile = open('train.txt', 'r')
data = [[],[]]
for line in trainFile:
    tmp=line.split()
    if tmp[2] == '1':
        data[0].append((float(tmp[0]),float(tmp[1])))
    elif tmp[2] == '2':
        data[1].append((float(tmp[0]),float(tmp[1])))


#set initial parameter theta(0) and iteration index k = 1
theta = [0]
k = 1
