#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

def quicksort(toSort):
    size = len(toSort)
    pivotIndex = 0
    storeIndex = pivotIndex+1

    sortedLeft = []
    sortedRight = []

    for i in range(size):
        if toSort[i] < toSort[pivotIndex]:
            toSort[i], toSort[storeIndex] = toSort[storeIndex], toSort[i]
            storeIndex += 1
    toSort[pivotIndex], toSort[storeIndex-1] = toSort[storeIndex - 1], toSort[pivotIndex]

    pivotIndex = storeIndex-1

    if len(toSort[:pivotIndex]):
        sortedLeft = quicksort(toSort[:pivotIndex])
    if len(toSort[pivotIndex+1:]):
        sortedRight = quicksort(toSort[pivotIndex+1:])
    return sortedLeft + [toSort[pivotIndex]] + sortedRight


toSort = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
result = quicksort(toSort)
print(result)

