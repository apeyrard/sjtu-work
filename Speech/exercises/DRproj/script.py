#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import subprocess
import argparse

def createGrammar(path):
    with open("gram", 'w+') as f:
        f.write("$digit = one | two | three | four | five | six | seven | eight | nine | oh | zero;\n")
        f.write("( sil ( [<$digit sp>] $digit ) sil )\n")

def createMonophones():
    with open("monophones1", 'w+') as f:
        txt = ("eight           eight\nfive            five\nfour            four\nnine            nine\noh              oh\none             one\nseven           seven\nsil             sil\nsix             six\nsp              sp\nthree           three\ntwo             two\nzero            zero\n")
        f.write(txt)
    with open("monophones0", 'w+') as f:
        txt = ("eight           eight\nfive            five\nfour            four\nnine            nine\noh              oh\none             one\nseven           seven\nsil             sil\nsix             six\nthree           three\ntwo             two\nzero            zero\n")
        f.write(txt)

def createMacro(folder, proto):
    with open(''.join([folder, "/macros"]),"w+") as f:
        with open(proto,'r') as source:
            line = source.readline()
            f.write(line)
        with open(''.join([folder,"/vFloors"]),'r') as source:
            f.write("~v \"varFloor1\"\n")
            source.readline()
            f.write(source.readline())
            f.write(source.readline())

def createHmmdefs(folder):
    with open(''.join([folder, "/hmmdefs"]), "w+") as f:
        with open(''.join([folder, "/proto"]), 'r') as proto:
            txt = ""
            #Skip first 4 lines
            for i in range(4):
                proto.readline()
            for line in proto:
                txt += line
        with open("monophones0", 'r') as words:
            for line in words:
                f.write("~h \"" + line.split()[-1]+"\"\n")
                f.write(txt)

def addSp(nb):
    # adding transitions to sil HMM
    src = "hmm" + str(nb)
    dest = "hmm" + str(nb+1)
    mkdir(dest)
    subprocess.call(["cp", src + "/macros", dest+"/macros"])
    subprocess.call(["cp", src + "/hmmdefs", dest+"/hmmdefs"])
    with open(dest+"/hmmdefs", 'a') as f:
        f.write("~h \"sp\"\n")
        #f.write("<BEGINHMM>\n")
        #f.write("<NUMSTATES> 3\n")
        #f.write("<STATE> 2\n")
        with open(src+"/hmmdefs", 'r') as source:
            while(True):
                if source.readline().split()[-1] == "\"sil\"": # find sil model
                    break
            #while(True):
                #if source.readline() == "<STATE> 3\n": #find center state
                    #break
            #while(True):
                #line = source.readline()
                #if line == "<STATE> 4\n":
                    #break
                #else:
                    #f.write(line)
        #f.write("<TRANSP> 3\n")
        #f.write("0.0 0.6 0.4\n")
        #f.write("0.0 0.6 0.4\n")
        #f.write("0.0 0.0 0.0\n")
        #f.write("<ENDHMM>\n")
            while(True):
                line = source.readline()
                if line.split()[0] == "~h":
                    break
                else:
                    f.write(line)

    # create sil.hed
    with open("sil.hed", 'w+') as f:
        f.write("AT 2 4 0.2 {sil.transP}\n")
        f.write("AT 4 2 0.2 {sil.transP}\n")
        f.write("AT 1 3 0.3 {sp.transP}\n")
        f.write("TI silst {sil.state[3],sp.state[2]}\n")

    src = "hmm" + str(nb+1)
    dest = "hmm" + str(nb+2)
    mkdir(dest)
    subprocess.call(["bin/HHEd", "-H", src+"/macros", "-H", src+"/hmmdefs", "-M", dest, "sil.hed", "monophones1"])

def HParse(gramPath, outPath):
    subprocess.call(["./bin/HParse", gramPath, outPath])

def mkdir(path):
    subprocess.call(["mkdir","-p",path])

def HCopy(mapPath):
    subprocess.call(["./bin/HCopy","-C", "cfgs/config_hcopy", "-S", mapPath])

def HCompV(scpPath, outFolder, proto):
    subprocess.call(["./bin/HCompV", "-C", "cfgs/config_tr", "-f", "0.01", "-m", "-S", scpPath, "-M", outFolder, proto])

def HERest(mlfPath, scpPath, source, dest, dictionnary):
    subprocess.call(["bin/HERest", "-C", "cfgs/config_tr", "-I", mlfPath, "-t", "250.0", "150.0", "1000.0", "-S", scpPath, "-H", source + "/macros", "-H", source + "/hmmdefs", "-M", dest, dictionnary])

def HVite(src, scpPath, mlfOut, wdnet, dictionnary, wordlist):
    subprocess.call(["bin/HVite", "-C", "cfgs/config_tr", "-H", src+"/macros", "-H", src+"/hmmdefs", "-S", scpPath, "-l", "'*'", "-i", mlfOut, "-w", wdnet, "-p", "0.0", "-s", "5.0", dictionnary, wordlist])

def HResults(refPath, wordlist, mlfPath):
    subprocess.call(["bin/HResults", "-I", refPath, wordlist, mlfPath])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Digit recognition')
    parser.add_argument('nbEval')
    parser.add_argument('nbReeval')

    args = parser.parse_args()

    # First : data preparation
    # Grammar is written in the gram file using the provided HTK high level language
    createGrammar("gram")

    HParse("gram","wdnet")
    # Thus creating the word network in the file wdnet

    # feature extraction
    # using HCopy
    # The mapping file were modified
    # I added the relative position of the folders
    mkdir("mfcc/train")

    HCopy("./mapping/train.mapping")

    mkdir("hmm0")
    HCompV("train.scp", "hmm0", "proto")

    # We then need to create the file monophone0 and monophone1
    createMonophones()

    # We also need to create macros and hmmdefs
    createMacro("hmm0", "proto")
    createHmmdefs("hmm0")

    # Reestimate nb times. nb > 0
    nb = int(args.nbEval)
    for i in range(nb):
        source = "hmm" + str(i)
        dest = "hmm" + str(i+1)
        mkdir(dest)
        HERest("label/train.nosp.mlf", "train.scp", source, dest, "monophones0")

    addSp(nb)

    # Reestimate add times. add > 0
    add = int(args.nbReeval)
    for i in range(add):
        source = "hmm" + str(nb+i+2)
        dest = "hmm" + str(nb+i+3)
        mkdir(dest)
        HERest("label/train.all.mlf", "train.scp", source, dest, "monophones1")

    # Testing
    mkdir("mfcc/dev")
    HCopy("mapping/dev.mapping")

    src = "hmm" + str(nb + add + 2)
    HVite(src, "dev.scp", "recout.mlf", "wdnet", "monophones1", "word.list")

    # verif
    HResults("label/dev.ref.mlf", "word.list", "recout.mlf")
