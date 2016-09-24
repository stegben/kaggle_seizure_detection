import sys
import glob
from pprint import pprint

import numpy as np
import scipy.io as sio
import pandas as pd

import joblib # joblib.dump(data, fname)
from joblib import Memory
from joblib import Parallel, delayed


CACHE_DIR = '../cache'
RAW_DATA_DIR = '../raw_data/'
TRAIN_RAW_MEMMAP_FNAME = '../train.npy'
TEST_RAW_MEMMAP_FNAME = '../test.npy'

memory = Memory(cachedir=CACHE_DIR, mmap_mode='w+', verbose=1)
memory_rom = Memory(cachedir=CACHE_DIR+'_rom', mmap_mode='r', verbose=1)


@memory.cache
def get_all_file_name(path_pattern):
    all_fpath = []
    for fpath in glob.glob(path_pattern):
        all_fpath.append(fpath)
    return len(all_fpath), all_fpath


def process_raw_train(data, idx, fpath):
    mat = sio.loadmat(fpath)
    data[idx, :, :] = mat['dataStruct']['data'][0][0]
    file_name = fpath.split('/')[-1]
    patient_id = int(file_name.split('_')[0])
    label = int(file_name.split('.')[0].split('_')[2])
    hour = -1 if label == 0 else (int(file_name.split('_')[1]) - 1) // 6
    return [patient_id, label, hour, file_name]


def process_raw_test(data, idx, fpath):
    mat = sio.loadmat(fpath)
    data[idx, :, :] = mat['dataStruct']['data'][0][0]
    file_name = fpath.split('/')[-1]
    patient_id = int(file_name.split('_')[0])
    return [patient_id, file_name]


@memory.cache
def extract_train_from_path(path_pattern):
    n_samples, all_fpath = get_all_file_name(path_pattern)
    data = np.memmap(TRAIN_RAW_MEMMAP_FNAME, shape=(n_samples ,240000, 16), dtype='float32', mode='w+')
    # data = np.zeros(shape=(n_samples ,240000, 16), dtype='float32')
    result = Parallel(n_jobs=24, verbose=12)([delayed(process_raw_train)(data, i, fpath) for i, fpath in enumerate(all_fpath)])

    train_result = {}
    train_result['data'] = data
    train_result['patient_id'] = [rec[0] for rec in result]
    train_result['label'] = [rec[1] for rec in result]
    train_result['hour'] = [rec[2] for rec in result]
    train_result['file_name'] = [rec[3] for rec in result]
    return train_result


@memory.cache
def extract_test_from_path(path_pattern):
    n_samples, all_fpath = get_all_file_name(path_pattern)
    data = np.memmap(TEST_RAW_MEMMAP_FNAME, shape=(n_samples ,240000, 16), dtype='float32', mode='w+')
    # data = np.zeros(shape=(n_samples ,240000, 16), dtype='float32')
    result = Parallel(n_jobs=24, verbose=12)([delayed(process_file_get_meta)(data, i, fpath) for i, fpath in enumerate(all_fpath)])

    test_result = {}
    test_result['data'] = data
    test_result['patient_id'] = [rec[0] for rec in result]
    test_result['file_name'] = [rec[1] for rec in result]
    return test_result

@memory.cache
def mean_of_all():
    pass

def gen_feature(raw_data_train, raw_data_test):
    pass


def main():
    train_data_path_pattern = sys.argv[1]
    test_data_path_pattern = sys.argv[2]

    raw_data_train = extract_train_from_path(train_data_path_pattern)
    print(raw_data_train.shape)
    raw_data_test = extract_test_from_path(test_data_path_pattern)
    print(raw_data_test.shape)

    train, test = gen_feature(raw_data_train, raw_data_test)

    data = {}
    data['train'] = train
    data['test'] = test

    fname = 'data.pkl'
    joblib.dump(fname, data)


if __name__ == '__main__':
    main()


'''
final data generated:
{
    train: DataFrame(n_samples, features + n extra info columns)
    test: DataFrame(n_samples, features)
}
'''
