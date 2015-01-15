#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import subprocess
import numpy as np

if __name__ == "__main__":
    m = 100
    n = 100
    mat = np.zeros((m,n), dtype=float)
    print(mat)
    for i in range(m):
        for j in range(n):
            args = ["./script.py", str(i), str(j)]
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            output = popen.stdout.read()
            result = float(str(output.split()[-8]).split('=')[-1][0:-2])
            mat[i, j] = result
    print(mat)
    np.savetxt("csv", mat)

