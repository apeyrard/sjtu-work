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

    call(["./bin/HCompV", "-C", "cfgs/config_tr", "-f", "0.01", "-m", "-S", "train.scp", "-M", "hmm0", "proto"])

    # We then need to create the file monophone0 and monophone1
    with open("monophones1", 'w+') as f:
        txt = ("eight           eight\nfive            five\nfour            four\nnine            nine\noh              oh\none             one\nseven           seven\nsil             sil\nsix             six\nsp              sp\nthree           three\ntwo             two\nzero            zero\n")
        f.write(txt)
    with open("monophones0", 'w+') as f:
        txt = ("eight           eight\nfive            five\nfour            four\nnine            nine\noh              oh\none             one\nseven           seven\nsil             sil\nsix             six\nthree           three\ntwo             two\nzero            zero\n")
        f.write(txt)


    # We also need to create macros and hmmdefs
    with open("hmm0/macros","w+") as f:
        with open("proto",'r') as source:
            line = source.readline()
            f.write(line)
        with open("hmm0/vFloors",'r') as source:
            f.write("~v \"varFloor1\"\n")
            source.readline()
            f.write(source.readline())
            f.write(source.readline())
    with open("hmm0/hmmdefs", "w+") as f:
        with open("hmm0/proto", 'r') as proto:
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

    # Reestimate nb times. nb > 0
    nb = 3
    for i in range(nb):
        source = "hmm" + str(i)
        dest = "hmm" + str(i+1)
        call(["mkdir","-p",dest])
        call(["bin/HERest", "-C", "cfgs/config_tr", "-I", "label/train.nosp.mlf", "-t", "250.0", "150.0", "1000.0", "-S", "train.scp", "-H", source + "/macros", "-H", source + "/hmmdefs", "-M", dest, "monophones0"])

    # adding transitions to sil HMM
    src = "hmm" + str(nb)
    dest = "hmm" + str(nb+1)
    call(["mkdir","-p",dest])
    call(["cp", src + "/macros", dest+"/macros"])
    call(["cp", src + "/hmmdefs", dest+"/hmmdefs"])
    with open(dest+"/hmmdefs", 'a') as f:
        f.write("~h \"sp\"\n")
        f.write("<BEGINHMM>\n")
        f.write("<NUMSTATES> 3\n")
        f.write("<STATE> 2\n")
        with open(src+"/hmmdefs", 'r') as source:
            while(True):
                if source.readline().split()[-1] == "\"sil\"": # find sil model
                    break
            while(True):
                if source.readline() == "<STATE> 3\n": #find center state
                    break
            while(True):
                line = source.readline()
                if line == "<STATE> 4\n":
                    break
                else:
                    f.write(line)
        f.write("<TRANSP> 3\n")
        f.write("1.0 0.0 0.0\n")
        f.write("0.0 1.0 0.0\n")
        f.write("0.0 0.0 1.0\n")
        f.write("<ENDHMM>\n")

    # create sil.hed
    with open("sil.hed", 'w+') as f:
        f.write("AT 2 4 0.2 {sil.transP}\n")
        f.write("AT 4 2 0.2 {sil.transP}\n")
        f.write("AT 1 3 0.3 {sp.transP}\n")
        f.write("TI silst {sil.state[3],sp.state[2]}\n")

    src = "hmm" + str(nb+1)
    dest = "hmm" + str(nb+2)
    call(["mkdir","-p",dest])
    call(["bin/HHEd", "-H", src+"/macros", "-H", src+"/hmmdefs", "-M", dest, "sil.hed", "monophones1"])


    # Reestimate add times. add > 0
    add = 2
    for i in range(add):
        source = "hmm" + str(nb+i+2)
        dest = "hmm" + str(nb+i+3)
        call(["mkdir","-p",dest])
        call(["bin/HERest", "-C", "cfgs/config_tr", "-I", "label/train.nosp.mlf", "-t", "250.0", "150.0", "1000.0", "-S", "train.scp", "-H", source + "/macros", "-H", source + "/hmmdefs", "-M", dest, "monophones0"])

