import os
import sys
import glob

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.cross_validation import LabelShuffleSplit
from sklearn.cross_validation import ShuffleSplit

from tqdm import tqdm

import joblib # joblib.dump(data, fname)
from joblib import Memory
from joblib import Parallel, delayed

from utils import bin_power
from utils import get_all_file_name


CACHE_DIR = '../nn_cache'
RAW_DATA_DIR = '../raw_data/'
SAFE_DATA_LABEL = '../raw_data/train_and_test_data_labels_safe.csv'
FRAME_SIZE = 400 * 10
FRAME_SPACING = 400 * 5
USE_POWER = False


def gen_metadata(fpath):
    file_name = fpath.split('/')[-1]
    if 'new' in file_name:
        file_name = file_name[4:]
    patient_id = int(file_name.split('_')[0])

    try:
        label = int(file_name.split('.')[0].split('_')[2])
        hour = -1 if label == 0 else (int(file_name.split('_')[1]) - 1) // 6
        return {
            'fname': file_name,
            'hour': hour,
            'label': label,
        }
    except:
        return {
            'fname': file_name,
        }


@memory.cache
def create_data(path_pattern, frame_size, spacing, is_test, use_power=True):
    n_samples, all_fpath = get_all_file_name(path_pattern, (not is_test))
    n_frames_per_sample = int((240000 - frame_size)/spacing + 1)
    n_channels = 16
    # freq_bin = [freq for freq in range(150)]
    freq_bin = [freq for freq in range(0, 30, 1)]
    freq_bin = freq_bin + [30, 35, 40, 50, 60, 80, 100, 120, 160, 199]
    n_freq_bins = len(freq_bin) - 1
    shape = [n_samples, n_channels, n_freq_bins, n_frames_per_sample]
    data = np.zeros(shape, dtype='float32')

    label = []
    hours = []
    fnames = []
    for idx, fpath in tqdm(enumerate(all_fpath)):
        print(fpath)
        # process features
        mat = sio.loadmat(fpath)
        raw_data = mat['dataStruct']['data'][0][0]

        for frame_idx in range(n_frames_per_sample):
            start = frame_idx * spacing
            end = frame_idx*spacing + frame_size
            raw_data_frame = raw_data[start:end, :]
            for channel_idx in range(n_channels):
                signal = raw_data_frame[:, channel_idx]
                if use_power:
                    power, power_ratio = bin_power(signal, Band=freq_bin, Fs=400)
                    power = np.nan_to_num(power)
                    data[idx, channel_idx, :, frame_idx] = power
                else:
                    power, power_ratio = bin_power(signal, Band=freq_bin, Fs=400)
                    power_ratio = np.nan_to_num(power_ratio)
                    data[idx, channel_idx, :, frame_idx] = power_ratio

        # process metadata
        if is_test:
            info = gen_metadata(fpath)
            fnames.append(info['fname'])
        else:
            info = gen_metadata(fpath)
            hours.append(info['hour'])
            fnames.append(info['fname'])
            label.append(info['label'])

    meta_data = {
        'fnames': fnames,
        'hours': np.array(hours, dtype=int),
    }
    label = np.array(label)
    return data, label, meta_data


def split_train(train_x, train_y, train_metadata):
    test_size=0.2
    random_state=5438

    train_x_true = train_x[train_y == 1]
    train_x_false = train_x[train_y != 1]
    train_y_true = train_y[train_y == 1]
    train_y_false = train_y[train_y != 1]

    hours = train_metadata['hours']
    hours_true = hours[train_y == 1]
    hours_false = hours[train_y != 1]

    assert -1 not in set(list(hours_true))
    assert set([-1]) == set(list(hours_false))

    lss = LabelShuffleSplit(hours_true, n_iter=1, test_size=test_size, random_state=random_state)
    true_idx = {}
    for tr_idx, te_idx in lss:
        true_idx['subtrain'] = tr_idx
        true_idx['validation'] = te_idx

    ss = ShuffleSplit(len(hours_false), n_iter=1, test_size=test_size, random_state=random_state)
    false_idx = {}
    for tr_idx, te_idx in ss:
        false_idx['subtrain'] = tr_idx
        false_idx['validation'] = te_idx

    subtrain_x = np.concatenate((train_x_true[true_idx['subtrain']], train_x_false[false_idx['subtrain']]))
    subtrain_y = np.concatenate((train_y_true[true_idx['subtrain']], train_y_false[false_idx['subtrain']]))
    validation_x = np.concatenate((train_x_true[true_idx['validation']], train_x_false[false_idx['validation']]))
    validation_y = np.concatenate((train_y_true[true_idx['validation']], train_y_false[false_idx['validation']]))
    return subtrain_x, subtrain_y, validation_x, validation_y


def main():
    train_1_data_path_pattern = '../raw_data/train_1/*.mat'
    train_2_data_path_pattern = '../raw_data/train_2/*.mat'
    train_3_data_path_pattern = '../raw_data/train_3/*.mat'
    test_1_data_path_pattern = '../raw_data/test_1_new/*.mat'
    test_2_data_path_pattern = '../raw_data/test_2_new/*.mat'
    test_3_data_path_pattern = '../raw_data/test_3_new/*.mat'

    output_data_name = '../cnn_' + sys.argv[1]

    print('create_data')
    train_1_x, train_1_y, train_1_metadata = create_data(train_1_data_path_pattern, FRAME_SIZE, FRAME_SPACING, False, USE_POWER)
    train_2_x, train_2_y, train_2_metadata = create_data(train_2_data_path_pattern, FRAME_SIZE, FRAME_SPACING, False, USE_POWER)
    train_3_x, train_3_y, train_3_metadata = create_data(train_3_data_path_pattern, FRAME_SIZE, FRAME_SPACING, False, USE_POWER)
    test_1_x, _, test_1_metadata = create_data(test_1_data_path_pattern, FRAME_SIZE, FRAME_SPACING, True)
    test_2_x, _, test_2_metadata = create_data(test_2_data_path_pattern, FRAME_SIZE, FRAME_SPACING, True)
    test_3_x, _, test_3_metadata = create_data(test_3_data_path_pattern, FRAME_SIZE, FRAME_SPACING, True)

    subtrain_x_1, subtrain_y_1, validation_x_1, validation_y_1 = split_train(train_1_x, train_1_y, train_1_metadata)
    subtrain_x_2, subtrain_y_2, validation_x_2, validation_y_2 = split_train(train_2_x, train_2_y, train_2_metadata)
    subtrain_x_3, subtrain_y_3, validation_x_3, validation_y_3 = split_train(train_3_x, train_3_y, train_3_metadata)
    # import ipdb; ipdb.set_trace()
    data = {
        'pid_1': {
            'subtrain_x': subtrain_x_1,
            'subtrain_y': subtrain_y_1,
            'validation_x': validation_x_1,
            'validation_y': validation_y_1,
            'test_x': test_1_x,
            'test_fnames': test_1_metadata['fnames'],
        },
        'pid_2': {
            'subtrain_x': subtrain_x_2,
            'subtrain_y': subtrain_y_2,
            'validation_x': validation_x_2,
            'validation_y': validation_y_2,
            'test_x': test_2_x,
            'test_fnames': test_2_metadata['fnames'],
        },
        'pid_3': {
            'subtrain_x': subtrain_x_3,
            'subtrain_y': subtrain_y_3,
            'validation_x': validation_x_3,
            'validation_y': validation_y_3,
            'test_x': test_3_x,
            'test_fnames': test_3_metadata['fnames'],
        },
    }

    joblib.dump(data, output_data_name)


if __name__ == '__main__':
    main()
