#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Choose how to process input image.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--hist', action='store_true', default=False, help='Draw image histogram')
group.add_argument('--enhance', action='store_true', default=False, help='Enhance image')

parser.add_argument('image', required=True)

parser.parse_args()
