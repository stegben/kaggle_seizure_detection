import sys
import glob
import datetime
from pprint import pprint

import numpy as np
import scipy.io as sio
from scipy.stats.stats import pearsonr
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import pandas as pd

import joblib # joblib.dump(data, fname)
from joblib import Memory
from joblib import Parallel, delayed

from sklearn.cross_validation import LabelShuffleSplit
from sklearn.cross_validation import ShuffleSplit

from utils import bin_power
from utils import spectrum_entropy

from utils import get_all_file_name

CACHE_DIR = '../cache'
RAW_DATA_DIR = '../raw_data/'
TRAIN_RAW_MEMMAP_FNAME = '../train_10_5.npy'
TEST_RAW_MEMMAP_FNAME = '../test_10_5.npy'

SPLIT = 10

FRAME_SIZE = 400 * 10
FRAME_SPACING = 400 * 5

memory = Memory(cachedir=CACHE_DIR, verbose=1)


def process_raw_train(data, idx, fpath, n_frames_per_sample, frame_size, spacing):
    mat = sio.loadmat(fpath)
    file_name = fpath.split('/')[-1]
    patient_id = int(file_name.split('_')[0])
    label = int(file_name.split('.')[0].split('_')[2])
    hour = -1 if label == 0 else (int(file_name.split('_')[1]) - 1) // 6

    pids = []
    labels = []
    hours = []
    fnames = []
    processed_array = np.ones((n_frames_per_sample, frame_size, 16), dtype='float32')
    for k in range(n_frames_per_sample):
        start = k * spacing
        end = k*spacing + frame_size
        processed_array[k, :, :] = mat['dataStruct']['data'][0][0][start:end, :]
        pids.append(patient_id)
        labels.append(label)
        hours.append(hour)
        fnames.append(file_name)
    start_idx = idx * n_frames_per_sample
    end_idx = (idx + 1) * n_frames_per_sample
    data[start_idx:end_idx, :, :] = processed_array

    return {
        'patient_id': pids,
        'label': labels,
        'hour': hours,
        'file_name': fnames,
    }


def process_raw_test(data, idx, fpath, n_frames_per_sample, frame_size, spacing):
    mat = sio.loadmat(fpath)
    file_name = fpath.split('/')[-1]
    if 'new' in file_name:
        file_name = file_name[4:]
    patient_id = int(file_name.split('_')[0])

    pids = []
    fnames = []
    processed_array = np.ones((n_frames_per_sample, frame_size, 16), dtype='float32')
    for k in range(n_frames_per_sample):
        start = k * spacing
        end = k*spacing + frame_size
        processed_array[k, :, :] = mat['dataStruct']['data'][0][0][start:end, :]
        pids.append(patient_id)
        fnames.append(file_name)
    start_idx = idx * n_frames_per_sample
    end_idx = (idx + 1) * n_frames_per_sample
    data[start_idx:end_idx, :, :] = processed_array

    return {
        'patient_id': pids,
        'file_name': fnames,
    }


@memory.cache
def extract_train_from_path(path_pattern, raw_memmap_path, frame_size, spacing):
    n_samples, all_fpath = get_all_file_name(path_pattern, is_train=True)
    # import ipdb; ipdb.set_trace()
    assert (240000 - frame_size) % spacing == 0
    n_frames_per_sample = int((240000 - frame_size)/spacing + 1)
    print('n_frames_per_sample: {}'.format(n_frames_per_sample))
    shape = (n_samples*n_frames_per_sample ,frame_size, 16)
    data = np.memmap(raw_memmap_path, shape=shape, dtype='float32', mode='w+')

    works = []
    for i, fpath in enumerate(all_fpath):
        works.append(delayed(process_raw_train)(data, i, fpath, n_frames_per_sample, frame_size, spacing))
    result = Parallel(n_jobs=2, verbose=1)(works)

    train_result = {}
    train_result['data'] = {'path': raw_memmap_path, 'shape': shape}
    train_result['patient_id'] = [pid for rec in result for pid in rec['patient_id']]
    train_result['label'] = [pid for rec in result for pid in rec['label']]
    train_result['hour'] = [pid for rec in result for pid in rec['hour']]
    train_result['file_name'] = [pid for rec in result for pid in rec['file_name']]
    del data
    return train_result


@memory.cache
def extract_test_from_path(path_pattern, raw_memmap_path, frame_size, spacing):
    n_samples, all_fpath = get_all_file_name(path_pattern, is_train=False)
    assert (240000 - frame_size) % spacing == 0
    n_frames_per_sample = int((240000 - frame_size)/spacing + 1)
    print('n_frames_per_sample: {}'.format(n_frames_per_sample))
    shape = (n_samples*n_frames_per_sample ,frame_size, 16)
    # import ipdb; ipdb.set_trace()
    data = np.memmap(raw_memmap_path, shape=shape, dtype='float32', mode='w+')
    # import ipdb; ipdb.set_trace()
    works = []
    for i, fpath in enumerate(all_fpath):
        works.append(delayed(process_raw_test)(data, i, fpath, n_frames_per_sample, frame_size, spacing))
    result = Parallel(n_jobs=2, verbose=1)(works)

    test_result = {}
    test_result['data'] = {'path': raw_memmap_path, 'shape': shape}
    test_result['patient_id'] = [pid for rec in result for pid in rec['patient_id']]
    test_result['file_name'] = [pid for rec in result for pid in rec['file_name']]
    del data
    return test_result


def get_mean(data):
    return data.mean(axis=0)

def gen_mean_of_each_channel(data):
    '''
    Input:
        - data: nparraywith shape(n_samples, 240000, n_channels)
    Output:
        - feature: DataFrame(n_samples, n_channels)
    '''
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:].copy()
        else:
            partial_data = data[start_idx: end_idx].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_mean)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    # raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_mean)(data, n) for n in range(n_samples))
    column_names = ['mean_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return  feature


def get_std(data):
    return data.std(axis=0)

def gen_std_of_each_channel(data):
    '''
    Input:
        - data: nparraywith shape(n_samples, 240000, n_channels)
    Output:
        - feature: DataFrame(n_samples, n_channels)
    '''
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:].copy()
        else:
            partial_data = data[start_idx: end_idx].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_std)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    # raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_std)(data, n) for n in range(n_samples))
    column_names = ['std_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return  feature


def get_1d_std(data):
    return np.diff(data, axis=0).std(axis=0)

def gen_1d_std_of_each_channel(data):
    '''
    Input:
        - data: nparraywith shape(n_samples, 240000, n_channels)
    Output:
        - feature: DataFrame(n_samples, n_channels)
    '''
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:].copy()
        else:
            partial_data = data[start_idx: end_idx].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_1d_std)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    column_names = ['1d_std_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return feature


def get_2d_std(data):
    return np.diff(data, axis=0, n=2).std(axis=0)

def gen_2d_std_of_each_channel(data):
    '''
    Input:
        - data: nparraywith shape(n_samples, 240000, n_channels)
    Output:
        - feature: DataFrame(n_samples, n_channels)
    '''
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:, :, :].copy()
        else:
            partial_data = data[start_idx: end_idx, :, :].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_1d_std)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    column_names = ['2d_std_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return feature


def get_abs_mean(data):
    return np.absolute(data).mean(axis=0)

def gen_abs_mean_of_each_channel(data):
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:].copy()
        else:
            partial_data = data[start_idx: end_idx].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_abs_mean)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    # raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_abs_mean)(data, n) for n in range(n_samples))
    column_names = ['abs_mean_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return  feature


def get_freq(data, idx, band, sampling_rate):
    data_extract = data[idx, :, :]
    powers = []
    for channel in range(data_extract.shape[1]):
        power, power_ratio = bin_power(data_extract[:, channel], Band=band, Fs=sampling_rate)
        powers.append(np.concatenate((power_ratio, power)))
    return np.concatenate(powers, axis=0)

@memory.cache
def gen_freq_of_each_channel(data):
    '''
    band: list of frequencies
    '''
    n_channels = data.shape[2]
    n_samples = data.shape[0]

    band = [freq for freq in range(0, 20, 1)]
    band = band + [freq for freq in range(20, 80, 6)]
    band = band + [100, 150, 200]
    # band = [0, 1, 2, 4, 8, 12, 16, 20, 30, 40, 50, 80, 120]

    raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_freq)(data, n, band, 400) for n in range(n_samples))

    column_names = []
    for idx in range(n_channels):
        for band_idx in range(len(band) - 1):
            freq = band[band_idx]
            name = 'channel_' + str(idx) + 'freq_ratio_' + str(freq)
            column_names.append(name)
        for band_idx in range(len(band) - 1):
            freq = band[band_idx]
            name = 'channel_' + str(idx) + 'freq_power_' + str(freq)
            column_names.append(name)
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names).fillna(0.0)
    return  feature


def get_entropy(data, idx, band, sampling_rate):
    signal = data[idx, :, :]
    channels = signal.shape[1]
    feature = []
    for i in range(channels):
        spent = spectrum_entropy(signal[:, i], band=band, sampling_rate=sampling_rate)
        feature.append(spent)
    return np.array(feature)

@memory.cache
def gen_entropy_of_each_channel(data):
    n_channels = data.shape[2]
    n_samples = data.shape[0]

    band = [freq for freq in range(0, 60, 2)]
    # band = [0, 1, 2, 4, 8, 12, 16, 20, 30, 40, 50, 80, 120]

    raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_entropy)(data, n, band, 400) for n in range(n_samples))
    column_names = ['entropy_of_channel_' + str(idx) for idx in range(n_channels)]
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    feature = feature.fillna(feature.mean())
    return  feature


def get_corr(data, idx):
    signals = data[idx, :, :]
    n_channels = data.shape[2]
    feature = []
    for k in range(n_channels):
        for l in range(k+1, n_channels):
            feature.append(pearsonr(signals[:, k], signals[:, l])[0])
    return np.array(feature)

@memory.cache
def gen_corr_between_signals(data):
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_corr)(data, n) for n in range(n_samples))

    column_names = []
    for k in range(n_channels):
        for l in range(k+1, n_channels):
            column_names.append('corr_between_{}_{}'.format(k, l))
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    feature = feature.fillna(0)
    return  feature


def get_freq_corr(data, idx, band, sampling_rate):
    signals = data[idx, :, :]
    n_channels = data.shape[2]

    signal_spectrums = []
    signal_spectrums_ratio = []
    for chl in range(n_channels):
        power, power_ratio = bin_power(signals[:, chl], band, sampling_rate)
        signal_spectrums.append(power)
        signal_spectrums_ratio.append(power_ratio)

    feature = []
    for k in range(n_channels):
        for l in range(k+1, n_channels):
            spectrum1 = signal_spectrums[k]
            spectrum2 = signal_spectrums[l]
            feature.append(pearsonr(spectrum1, spectrum2)[0])
            spectrum1 = signal_spectrums_ratio[k]
            spectrum2 = signal_spectrums_ratio[l]
            feature.append(pearsonr(spectrum1, spectrum2)[0])
    return np.array(feature)

@memory.cache
def gen_freq_corr_between_signals(data):
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    band = [freq for freq in range(0, 100)]
    raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_freq_corr)(data, n, band, 400) for n in range(n_samples))

    column_names = []
    for k in range(n_channels):
        for l in range(k+1, n_channels):
            column_names.append('freq_corr_between_{}_{}'.format(k, l))
            column_names.append('freq_corr_between_{}_{}_ratio'.format(k, l))
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    feature = feature.fillna(0)
    return  feature


def get_skew_kurtosis(data):
    sk = skew(data)
    kurt = kurtosis(data)
    return np.concatenate((sk, kurt))

# @memory.cache
def gen_skew_kurtosis(data):
    n_channels = data.shape[2]
    n_samples = data.shape[0]
    # raw_feature = Parallel(n_jobs=2, verbose=1)(delayed(get_skew_kurtosis)(data, n) for n in range(n_samples))
    raw_feature = []
    for idx in range(SPLIT):
        print('in batch {}'.format(idx))
        start_idx = idx * (n_samples // SPLIT)
        end_idx = (idx + 1) * (n_samples // SPLIT)
        if idx == SPLIT - 1:
            partial_data = data[start_idx:, :, :].copy()
        else:
            partial_data = data[start_idx: end_idx, :, :].copy()
        partial_raw_feature = Parallel(n_jobs=6, verbose=1)(delayed(get_skew_kurtosis)(partial_data[n, :, :]) for n in range(partial_data.shape[0]))
        raw_feature = raw_feature + partial_raw_feature
        del partial_data
    column_names = []
    for idx in range(n_channels):
        column_names.append('skew_channel' + str(idx))
        column_names.append('kurtosis_channel' + str(idx))
    # import ipdb; ipdb.set_trace()
    feature = pd.DataFrame(np.array(raw_feature), columns=column_names)
    return  feature


@memory.cache
def gen_label(train_labels):
    return pd.Series(train_labels, name='label')

@memory.cache
def gen_file_name(fnames):
    return pd.Series(fnames, name='fnames', dtype=str)

@memory.cache
def gen_patient_id(pids):
    return pd.Series(pids, name='patient_id', dtype=str)

@memory.cache
def gen_dummy_pid(train_pid, test_pid):
    df_train = pd.DataFrame({'pid': train_pid})
    df_train['is_test'] = False

    df_test = pd.DataFrame({'pid': test_pid})
    df_test['is_test'] = True

    df = pd.concat([df_train, df_test], axis=0)
    new_df = pd.get_dummies(df, columns=['pid'], prefix='pid')

    new_df_train = new_df[~new_df['is_test']].drop('is_test', axis=1)
    new_df_test = new_df[new_df['is_test']].drop('is_test', axis=1)
    return new_df_train, new_df_test


@memory.cache
def gen_hour(hours):
    return pd.Series(hours, name='hour', dtype=int)

@memory.cache
def gen_feature(raw_data_train, raw_data_test):

    # generate some meta column of training set
    print('generate meta column for training set')
    train_file_name = gen_file_name(raw_data_train['file_name'])
    train_patient_id = gen_patient_id(raw_data_train['patient_id'])
    label = gen_label(raw_data_train['label'])
    hour = gen_hour(raw_data_train['hour'])

    # generate some meta column of testing set
    print('generate meta column for testing set')
    test_file_name = gen_file_name(raw_data_test['file_name'])
    test_patient_id = gen_patient_id(raw_data_test['patient_id'])

    # get dummy patient id
    train_dummy_pid, test_dummy_pid = gen_dummy_pid(train_patient_id, test_patient_id)

    # gen feature
    print('generate feature of each channel of train...')
    train_array = np.memmap(raw_data_train['data']['path'], shape=raw_data_train['data']['shape'], dtype='float32', mode='r')
    train_1d_std_of_each_channel = gen_1d_std_of_each_channel(train_array)
    train_2d_std_of_each_channel = gen_2d_std_of_each_channel(train_array)
    train_skew_kurtosis = gen_skew_kurtosis(train_array)
    train_mean_of_each_channel = gen_mean_of_each_channel(train_array)
    train_std_of_each_channel = gen_std_of_each_channel(train_array)
    train_abs_mean_of_each_channel = gen_abs_mean_of_each_channel(train_array)
    train_freq_of_each_channel = gen_freq_of_each_channel(train_array)
    train_entropy_of_each_channel = gen_entropy_of_each_channel(train_array)
    train_corr_between_signals = gen_corr_between_signals(train_array)
    train_freq_corr_between_signals = gen_freq_corr_between_signals(train_array)

    print('generate feature of each channel of test...')
    test_array = np.memmap(raw_data_test['data']['path'], shape=raw_data_test['data']['shape'], dtype='float32', mode='r')
    test_mean_of_each_channel = gen_mean_of_each_channel(test_array)
    test_std_of_each_channel = gen_std_of_each_channel(test_array)
    test_abs_mean_of_each_channel = gen_abs_mean_of_each_channel(test_array)
    test_1d_std_of_each_channel = gen_1d_std_of_each_channel(test_array)
    test_2d_std_of_each_channel = gen_2d_std_of_each_channel(test_array)
    test_freq_of_each_channel = gen_freq_of_each_channel(test_array)
    test_entropy_of_each_channel = gen_entropy_of_each_channel(test_array)
    test_corr_between_signals = gen_corr_between_signals(test_array)
    test_freq_corr_between_signals = gen_freq_corr_between_signals(test_array)
    test_skew_kurtosis = gen_skew_kurtosis(test_array)

    df_train = pd.concat([
        train_file_name,
        train_mean_of_each_channel,
        train_std_of_each_channel,
        train_1d_std_of_each_channel,
        train_2d_std_of_each_channel,
        train_abs_mean_of_each_channel,
        train_freq_of_each_channel,
        train_entropy_of_each_channel,
        train_corr_between_signals,
        train_freq_corr_between_signals,
        train_skew_kurtosis,
        train_dummy_pid,
        label,
        hour,
    ], axis=1)
    df_test = pd.concat([
        test_file_name,
        test_mean_of_each_channel,
        test_std_of_each_channel,
        test_1d_std_of_each_channel,
        test_2d_std_of_each_channel,
        test_abs_mean_of_each_channel,
        test_freq_of_each_channel,
        test_entropy_of_each_channel,
        test_corr_between_signals,
        test_freq_corr_between_signals,
        test_skew_kurtosis,
        test_dummy_pid,
    ], axis=1)
    import ipdb; ipdb.set_trace()

    # del train_array
    # del test_array

    return  df_train, df_test


@memory.cache
def split_train_by_label(df_train, test_size=0.2, random_state=123):
    columns = df_train.columns

    df_train_false = df_train[df_train['hour'] == -1]
    df_train_true = df_train[df_train['hour'] != -1]

    lss = LabelShuffleSplit(df_train_true['hour'], n_iter=1, test_size=test_size, random_state=random_state)
    true_idx = {}
    for tr_idx, te_idx in lss:
        true_idx['subtrain'] = tr_idx
        true_idx['validation'] = te_idx

    ss = ShuffleSplit(df_train_false.values.shape[0], n_iter=1, test_size=test_size, random_state=random_state)
    false_idx = {}
    for tr_idx, te_idx in ss:
        false_idx['subtrain'] = tr_idx
        false_idx['validation'] = te_idx

    df_subtrain = pd.concat([
        df_train_true.iloc[true_idx['subtrain'], :],
        df_train_false.iloc[false_idx['subtrain'], :],
    ], axis=0)

    df_validation = pd.concat([
        df_train_true.iloc[true_idx['validation'], :],
        df_train_false.iloc[false_idx['validation'], :],
    ], axis=0)
    return df_subtrain, df_validation


def main():
    train_data_path_pattern = sys.argv[1]
    test_data_path_pattern = sys.argv[2]
    output_data_name = '../' + datetime.datetime.now().strftime("%Y%m%d%H") + sys.argv[3]

    raw_data_train = extract_train_from_path(train_data_path_pattern, TRAIN_RAW_MEMMAP_FNAME, frame_size=FRAME_SIZE, spacing=FRAME_SPACING)
    raw_data_test = extract_test_from_path(test_data_path_pattern, TEST_RAW_MEMMAP_FNAME, frame_size=FRAME_SIZE, spacing=FRAME_SPACING)

    df_train, df_test = gen_feature(raw_data_train, raw_data_test)

    df_subtrain, df_validation = split_train_by_label(df_train)
    df_subtrain = df_subtrain.drop('hour', axis=1)
    df_validation = df_validation.drop('hour', axis=1)

    split_data = {}
    split_data['subtrain'] = df_subtrain
    split_data['validation'] = df_validation
    split_data['test'] = df_test

    joblib.dump(split_data, output_data_name)


if __name__ == '__main__':
    main()


'''
final data generated:
{
    train: DataFrame(n_samples, features + extra info columns)
    test: DataFrame(n_samples, features)
}
'''
