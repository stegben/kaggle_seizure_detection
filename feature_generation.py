import sys
import glob

import numpy as np
import scipy.io as sio
from joblib import Memory
from tqdm import tqdm

CACHE_DIR = './cache'
memory = Memory(cachedir=CACHE_DIR, mmap_mode='w+')

@memory.cache
def get_all_file_name():


@memory.cache
def extract_train_from_path(path_pattern):
    all_fpath = []
    for fpath in glob.glob(path_pattern):

        all_fpath.append(fpath)

    n_samples = len(all_fpath)
    data = np.memmap(filename='temp.npy', shape=(n_samples ,240000, 16), dtype='float32', mode='w+')
    all_label = []
    all_hour = []
    all_patient_id = []
    all_file_name = []
    for i, fpath in tqdm(enumerate(all_fpath)):
        mat = sio.loadmat(fpath)
        data[i, :, :] = mat['dataStruct']['data'][0][0]

        file_name = fpath.split('/')[-1]
        patient_id = int(file_name.split('_')[0])
        label = int(file_name.split('.')[0].split('_')[2])
        hour = -1 if label == '0' else (int(file_name.split('_')[1]) - 1) / 6

        all_label.append(label)
        all_hour.append(hour)
        all_patient_id.append(patient_id)
        all_file_name.append(file_name)

    result = {}
    result['data'] = data
    result['label'] = np.array(all_label)
    result['hour'] = np.array(all_hour)
    result['patient_id'] = np.array(all_patient_id)
    result['file_name'] = all_file_name
    # return result
    return data


def main():
    train_data_path_pattern = sys.argv[1]
    print(train_data_path_pattern)
    test_data_path_pattern = sys.argv[2]

    data_train = extract_train_from_path(train_data_path_pattern)
    # data_test = extract_test_from_path(test_data_path_pattern)

    # label =
    # hour =
    # patient_id_train =
    # file_name_train =

    # patient_id_test =
    # file_name_test =

    # array_train =
    # array_test =


if __name__ == '__main__':
    main()
