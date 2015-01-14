#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from subprocess import call

if __name__ == "__main__":

    # First : data preparation
    # Grammar is written in the gram file using the provided HTK high level language
    call(["./bin/HParse","gram","wdnet"])
    # Thus creating the word network in the file wdnet


    # feature extraction
    # using HCopy
    # The mapping file were modified
    # I added the relative position of the folders
    call(["./bin/HCopy","-C", "./cfgs/config_hcopy", "-S", "./mapping/train.mapping"])
