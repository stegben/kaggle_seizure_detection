import sys
import glob

import numpy as np
import scipy.io as sio

from sklearn.cross_validation import LabelShuffleSplit
from sklearn.cross_validation import ShuffleSplit

from tqdm import tqdm

import joblib # joblib.dump(data, fname)
from joblib import Memory
from joblib import Parallel, delayed

from utils import bin_power


CACHE_DIR = '../nn_cache'
RAW_DATA_DIR = '../raw_data/'
FRAME_SIZE = 400 * 9
FRAME_SPACING = 400 * 3

memory = Memory(cachedir=CACHE_DIR, verbose=1)


@memory.cache
def get_all_file_name(path_pattern):
    all_fpath = []
    for fpath in glob.glob(path_pattern):
        all_fpath.append(fpath)
    return len(all_fpath), all_fpath


def gen_metadata(fpath):
    file_name = fpath.split('/')[-1]
    patient_id = int(file_name.split('_')[0])
    try:
        label = int(file_name.split('.')[0].split('_')[2])
        hour = -1 if label == 0 else (int(file_name.split('_')[1]) - 1) // 6
        return {
            'fname': file_name,
            'hour': hour,
            'label': label,
            'patient_id': patient_id,
        }
    except:
        return {
            'fname': file_name,
            'patient_id': patient_id,
        }

'''
def process_single_mat(processed_data, frame_idx):
    start = frame_idx * spacing
    end = frame_idx*spacing + frame_size
    raw_data_frame = raw_data[start:end, :]
    for channel_idx in range(n_channels):
        signal = raw_data_frame[:, channel_idx]
        _, power_ratio = bin_power(signal, Band=freq_bin, Fs=400)
        data[channel_idx, :, frame_idx] = power_ratio
'''

@memory.cache
def create_data(path_pattern, frame_size, spacing, is_test):
    n_samples, all_fpath = get_all_file_name(path_pattern)
    n_frames_per_sample = int((240000 - frame_size)/spacing + 1)
    n_channels = 16
    freq_bin = [freq for freq in range(0, 200, 2)]
    n_freq_bins = len(freq_bin) - 1
    shape = [n_samples, n_channels, n_freq_bins, n_frames_per_sample]
    data = np.zeros(shape, dtype='float32')

    label = []
    hours = []
    fnames = []
    patient_id = []
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
                _, power_ratio = bin_power(signal, Band=freq_bin, Fs=400)
                power_ratio = np.nan_to_num(power_ratio)
                data[idx, channel_idx, :, frame_idx] = power_ratio
        '''
        processed_data = np.zeros((n_channels, n_freq_bins, n_frames_per_sample), dtype='float32')
        Parallel(n_jobs=4, verbose=1)(
            delayed(process_single_mat)(processed_data, frame_idx) for frame_idx in range(n_frames_per_sample)
        )
        data[idx] = processed_data
        '''
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
        'patient_id': np.array(patient_id, dtype=int),
        'hours': np.array(hours, dtype=int),
    }
    label = np.array(label)
    return data, label, meta_data


def split_train(train_x, train_y, train_metadata):
    test_size=0.2
    random_state=123

    train_x_true = train_x[train_y == 1]
    train_x_false = train_x[train_y != 1]
    train_y_true = train_y[train_y == 1]
    train_y_false = train_y[train_y != 1]
    train_pid_true = train_metadata['patient_id'][train_y == 1]
    train_pid_false = train_metadata['patient_id'][train_y != 1]

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
    subtrain_pid = np.concatenate((train_pid_true[true_idx['subtrain']], train_pid_false[false_idx['subtrain']]))
    validation_x = np.concatenate((train_x_true[true_idx['validation']], train_x_false[false_idx['validation']]))
    validation_y = np.concatenate((train_y_true[true_idx['validation']], train_y_false[false_idx['validation']]))
    validation_pid = np.concatenate((train_pid_true[true_idx['validation']], train_pid_false[false_idx['validation']]))

    subtrain_metadata = {
        'patient_id': subtrain_pid,
    }
    validation_metadata = {
        'patient_id': validation_pid,
    }

    import ipdb; ipdb.set_trace()
    return subtrain_x, subtrain_y, subtrain_metadata, validation_x, validation_y, validation_metadata


def main():
    train_data_path_pattern = '../raw_data/train_*/*.mat'
    test_data_path_pattern = '../raw_data/test_*/*.mat'

    output_data_name = '../cnn_' + sys.argv[1]

    print('create_data')
    train_x, train_y, train_metadata = create_data(train_data_path_pattern, FRAME_SIZE, FRAME_SPACING, False)
    test_x, _, test_metadata = create_data(test_data_path_pattern, FRAME_SIZE, FRAME_SPACING, True)
    subtrain_x, subtrain_y, subtrain_metadata, validation_x, validation_y, validation_metadata = \
        split_train(train_x, train_y, train_metadata)

    import ipdb; ipdb.set_trace()
    data = {
        'pid': {
            'subtrain_x': subtrain_x,
            'subtrain_y': subtrain_y,
            'validation_x': validation_x,
            'validation_y': validation_y,
            'test_x': test_x,
            'test_fnames': test_metadata['fnames'],
            'subtrain_pid': subtrain_metadata['patient_id'],
            'validation_pid': validation_metadata['patient_id'],
            'test_pid': test_metadata['patient_id'],
        }
    }

    joblib.dump(data, output_data_name)


if __name__ == '__main__':
    main()
