#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='Choose how to process input image.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--hist', action='store_true', default=False, help='Draw image histogram')
group.add_argument('--enhance', action='store_true', default=False, help='Enhance image')

parser.add_argument('image')

args =  parser.parse_args()

try:
    im = Image.open(args.image)
except FileNotFoundError as e:
    sys.exit("Error : file not found")

data = list(im.getdata())
hist = [0] * 256
for pixel in data:
    if not (pixel[0] == pixel[1] == pixel[2]):
        sys.exit("Error : Image is not grayscale")
    else:
        hist[pixel[0]] += 1


