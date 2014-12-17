#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
import random
from PIL import Image

parser = argparse.ArgumentParser(description='Adds uniform noise to image.')

parser.add_argument('image')
parser.add_argument('outfile')
parser.add_argument('mu')
parser.add_argument('variance')

args =  parser.parse_args()

try:
    im = Image.open(args.image)
except FileNotFoundError as e:
    sys.exit("Error : file not found")

data = list(im.getdata())


newim = Image.new(im.mode, im.size)
ptr = newim.load()

for i, pixel in enumerate(data):
    newValue = pixel + random.gauss(float(args.mu),float(args.variance)**0.5)
    ptr[i%im.size[0], i//im.size[0]] = newValue
newim.save(args.outfile)

