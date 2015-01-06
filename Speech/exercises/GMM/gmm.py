#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def gauss(x, mu, sigma):
    D = len(x)
    diff = (x - mu)
    det = np.linalg.det(sigma)
    norm = 1/(math.pow((2*np.pi),float(D)/2) * math.pow(det,1.0/2) )
    inv = sigma.I
    result = math.pow(math.e, -0.5*(diff.T * inv * diff)) * norm
    return result

def kmeans(data, K, D):
    restart = True
    while restart:
        print("Initializing values with k-means")
        # K-means

        # Random starting points
        means = (data.max() - data.min()) * np.random.random((K,D)) + data.min()
        assign = np.zeros((len(data)), dtype=int)

        it = 0
        while(True):
            it += 1
            # Expectation step
            oldAssign = assign.copy()
            for j,point in enumerate(data):
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

            # End criterion
            if not (oldAssign - assign).any():
                break

            # Maximization step
            for i,mean in enumerate(means):
                num = np.array([0, 0], dtype=float)
                denom = 0
                for j,point in enumerate(data):
                    if assign[j] == i:
                        num += point
                        denom +=1
                if denom != 0:
                    means[i] = num/denom
                else:
                    means[i] = (data.max() - data.min()) * np.random.random(D) + data.min()

        # Plot K-means result for human verification
        print("K-means finished in", it, "iterations")
        plt.figure(1)
        plt.scatter(data[:,0], data[:,1], c=assign)
        plt.scatter(means[:,0], means[:,1], c=u'g')

        plt.ion() # Interactive mode : don't stop while showing
        plt.show()

        print("Is result correct Y/n ?")
        inp = str(input())
        plt.close()
        if inp == 'y' or inp == '' or inp == 'Y':
            restart = False
        else:
            print("Incorrect result, restarting K-means")
    return means, assign


def gmm(data, K, D, means=None, covars=None, pi=None):
    # GMM
    print("\nStarting fitting GMM\n")

    # If init means ar not given, they are random
    if means is None:
        means = (data.max() - data.min()) * np.random.random((K,D)) + data.min()

    # Covariances
    if covars is None:
        covars = []
        for i in range(K):
            a = np.array([[1, 0],[0, 1]])
            covars.append(a)

    # Mixing coefficients
    if pi is None:
        pi = np.zeros(K)
        for i in range(K):
            pi[i] = 1/K

    # Convergence criterion
    epsilon = 1e-8
    logL = 0
    while(True):
        # E step
        print("E step")
        # Responsibilities
        resp = np.zeros((len(data), K), dtype=float)
        for i,point in enumerate(data):
            pointMat = np.matrix(point).T
            denom = 0
            for j in range(K):
                denom += pi[j]*gauss(pointMat, np.matrix(means[j]).T,np.matrix(covars[j]))
            for cluster in range(K):
                meanMat = np.matrix(means[cluster]).T
                covarsMat = np.matrix(covars[cluster])
                num = pi[cluster]*gauss(pointMat, meanMat, covarsMat)
                resp[i, cluster] = num/denom

        Nk = []
        for i in range(K):
            Nk.append(sum(resp[:,i]))

        # M step
        print("M step")
        # New means
        for i in range(K):
            tmp = 0
            for n,point in enumerate(data):
                tmp += resp[n, i] * point
            means[i] = (1/Nk[i]) * tmp

        # New covariances
        for i in range(K):
            mean = np.matrix(means[i]).T
            tmp = 0
            for n,point in enumerate(data):
                point = np.matrix(point).T
                tmp += resp[n, i] * np.dot((point - mean), (point - mean).T)
            covars[i] = (1/Nk[i]) * tmp

        # New mixing coefs
        for i in range(K):
            pi[i] = Nk[i] / len(data)

        # Evaluate log likelihood
        newlogL = 0
        for n,point in enumerate(data):
            tmp = 0
            for k in range(K):
                tmp += pi[k] * gauss(np.matrix(point).T, np.matrix(means[k]).T, np.matrix(covars[k]))
            newlogL += np.log(tmp)
        print("Log likelihood : ", newlogL)
        if abs(newlogL - logL) < epsilon:
            break # criterion satisfied

        logL = newlogL
    return means, covars, pi

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Class prediction based on GMM')

    #parser.add_argument('--show', '-s', action = 'store_true')
    #parser.add_argument('--test', '-t', action = 'store_true')
    #parser.add_argument('--init', '-i', action = 'store_true')
    #parser.add_argument('training_data')
    #parser.add_argument

    #args = parser.parse_args()

    # load training data
    trainFile = open('train.txt', 'r')
    data = [[],[]]
    classes = []
    K = 4 # arbitrary, obtained by looking at data
    D = 2 # 2 dimensions

    for line in trainFile:
        tmp=line.split()
        data[ int(tmp[2])-1 ].append( (float(tmp[0]), float(tmp[1])) )

    for i in range(D):
        classes.append(np.array(data[i]))

    total = []
    for item in classes:
        total = np.append(total, item)

    # initialisation
    means = []
    assign = []

    print("Initialization")
    for i in range(D):
        a, b = kmeans(classes[i], K, D)
        means.append(a)
        assign.append(b)

    # Covariances
    covars = []
    clusters = []
    for j in range(D):
        a = []
        b = [[],[],[],[]]
        for n,point in enumerate(classes[j]):
            b[assign[j][n]].append(point)
        for i in range(K):
            a.append(np.cov(np.array(b[i]).T))
        covars.append(a)
        clusters.append(b)

    # Mixing coefficients
    pi = []
    for j in range(D):
        a = np.zeros(K)
        for i in range(K):
            a[i] = list(assign[j]).count(i) / len(assign[j])
        pi.append(a)

    for j in range(D):
        means[j], covars[j], pi[j] = gmm(classes[j], K, D, means[j], covars[j], pi[j])

        plt.figure(2)
        delta = 0.025
        x = np.arange(total.min(), total.max(), delta)
        y = np.arange(total.min(), total.max(), delta)
        X, Y = np.meshgrid(x, y)
        for i in range(K):
            Z = mlab.bivariate_normal(X, Y,np.sqrt(covars[j][i][0, 0]), np.sqrt(covars[j][i][1, 1]), means[j][i][0], means[j][i][1], covars[j][i][1,0])
            plt.contour(X,Y,Z)
        plt.scatter(classes[j][:,0], classes[j][:,1])


    # Starting prediction
    print("Starting prediction test")
    # load testing data
    nbOk = 0
    devFile = open('dev.txt', 'r')
    data = [[],[]]
    for line in devFile:
        tmp=line.split()
        data[0].append((float(tmp[0]),float(tmp[1])))
        data[1].append((int(tmp[2])))

    data = np.array(data)

    for n, point in enumerate(data[0]):
        pointMat = np.matrix(point[0]).T

        logs = []
        for j in range(D):
            tmp = 0
            for k in range(K):
                tmp += pi[j][k] * gauss(np.matrix(point).T, np.matrix(means[j][k]).T, np.matrix(covars[j][k]))
            logs.append(np.log(tmp))

        m = max(logs)
        pos = [n for n, o in enumerate(logs) if o == m][0]

        if pos == data[1, n]-1:
            nbOk += 1
            #plt.scatter(point[0], point[1], c=u'g')
        else:
            pass#plt.scatter(point[0], point[1], c=u'r')
    print("Nb corrects :", nbOk, "total :", len(data[0]))

    print("Starting prediction on real data")
    # load real data
    nbOk = 0
    with open('test.txt', 'r') as f:
        with open('out.txt','w') as out:
            data = []
            for line in f:
                tmp=line.split()
                data.append((float(tmp[0]),float(tmp[1])))

            data = np.array(data)

            for n, point in enumerate(data):
                pointMat = np.matrix(point[0]).T

                logs = []
                for j in range(D):
                    tmp = 0
                    for k in range(K):
                        tmp += pi[j][k] * gauss(np.matrix(point).T, np.matrix(means[j][k]).T, np.matrix(covars[j][k]))
                    logs.append(np.log(tmp))

                m = max(logs)
                pos = [n for n, o in enumerate(logs) if o == m][0]
                if pos == 0:
                    plt.scatter(point[0], point[1], c=u'r')
                else:
                    plt.scatter(point[0], point[1], c=u'g')
                out.write(str(point[0])+' '+str(point[1])+' '+str(pos+1)+'\n')

    plt.ioff() # Interctive mode off
    plt.show()
