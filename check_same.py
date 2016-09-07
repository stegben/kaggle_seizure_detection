import os
import sys
import glob

import scipy.io as sio


def main():
    all_samp_rate = set()
    for fname in glob.glob('./raw_data/*/*.mat'):
        mat = sio.loadmat(fname)
        samp_rate = mat['dataStruct']['iEEGsamplingRate'][0][0][0][0]
        all_samp_rate.add(samp_rate)



if __name__ == '__main__':
    main()
