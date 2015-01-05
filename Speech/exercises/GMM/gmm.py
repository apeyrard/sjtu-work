#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def gauss(x, mu, sigma):
    diff = (x - mu)
    return ((1/((2*np.pi)**(D/2)))
            *(1/(np.linalg.det(sigma)**0.5))
            *(math.exp(-0.5*(np.dot(np.dot(diff.T,sigma.I),diff)))))

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

# GMM

# initialisation

# The means are the means computed by k-means

# Covariances
covars = []
clusters = [[],[],[],[]]
for n,point in enumerate(class1):
    clusters[assign[n]].append(point)
for i in range(K):
    covars.append(np.cov(np.array(clusters[i]).T))


# Mixing coefficients
pi = np.zeros(K)
for i in range(K):
    pi[i] = list(assign).count(i) / len(assign)


# Convergence criterion
epsilon = 0.01
logL = 0
while(True):
    # E step

    # Responsibilities
    resp = np.zeros((len(class1), K), dtype=float)
    for cluster in range(K):
        for i,point in enumerate(class1):
            num = pi[cluster]*gauss(np.matrix(point).T,np.matrix(means[cluster]).T,np.matrix(covars[cluster]))
            denom = 0
            for j in range(K):
                denom += pi[j]*gauss(np.matrix(point).T,np.matrix(means[j]).T,np.matrix(covars[j]))
            resp[i, cluster] = num/denom

    # M step

    # New means
    for i in range(K):
        tmp = 0
        for n,point in enumerate(class1):
            tmp += resp[n, i] * point
        means[i] = (1/sum(resp[:,i])) * tmp

    # New covariances
    for i in range(K):
        mean = np.matrix(means[i]).T
        tmp = 0
        for n,point in enumerate(class1):
            point = np.matrix(point).T
            tmp += resp[n, i] * np.dot((point - mean), (point - mean).T)
        covars[i] = (1/sum(resp[:,i])) * tmp

    # New mixing coefs
    for i in range(K):
        pi[i] = sum(resp[:,i]) / len(class1)

    # Evaluate log likelihood
    newlogL = 0
    for n,point in enumerate(class1):
        tmp = 0
        for k in range(K):
            tmp += pi[k] * gauss(np.matrix(point).T, np.matrix(means[k]).T, np.matrix(covars[k]))
        newlogL += np.log(tmp)
    print("Log likelihood : ", newlogL)
    if abs(newlogL - logL) < epsilon:
        break # criterion satisfied

    logL = newlogL

delta = 0.025
x = np.arange(class1.min(), class1.max(), delta)
y = np.arange(class1.min(), class1.max(), delta)
X, Y = np.meshgrid(x, y)
for i in range(K):
    Z = mlab.bivariate_normal(X, Y,np.sqrt(covars[i][0, 0]), np.sqrt(covars[i][1, 1]), means[i][0], means[i][1], covars[i][1,0])
    plt.contour(X,Y,Z)
plt.scatter(class1[:,0], class1[:,1], c=assign)
plt.show()
