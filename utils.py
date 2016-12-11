import numpy as np
from joblib import Memory

CACHE_DIR = '../nn_cache'


def bin_power(X, Band, Fs):
    C = np.fft.rfft(X)
    C = np.absolute(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        # Power[Freq_Index] = np.sqrt(np.mean(np.square(
        #     C[np.floor(Freq * len(X) / Fs): np.floor(Next_Freq * len(X) / Fs)]
        # )))
        Power[Freq_Index] = np.sum(
            C[np.floor(Freq * len(X) / Fs): np.floor(Next_Freq * len(X) / Fs)]
        )

    power_sum = np.sum(Power)
    if power_sum <= 0:
        power_sum = 1e-10
    Power_Ratio = Power / np.sum(Power)
    return Power, Power_Ratio


def spectrum_entropy(signal, band, sampling_rate):
    power, power_ratio = bin_power(signal, band, sampling_rate)
    return np.sum(np.multiply(power_ratio, np.log(power_ratio)), axis=0)


memory = Memory(cachedir=CACHE_DIR, verbose=1)
@memory.cache
def get_all_file_name(path_pattern, is_train):
    df_correction = pd.read_csv(SAFE_DATA_LABEL)
    safe_train = set(df_correction[df_correction['safe'] == 1]['image'].tolist())
    all_fpath = []
    for fpath in glob.glob(path_pattern):
        if is_train:
            if os.path.basename(fpath) not in safe_train:
                print('drop: {} because unsafe'.format(fpath))
                continue
            if os.path.getsize(fpath) < 60001:
                print('drop: {} due to filesize'.format(fpath))
                continue
        all_fpath.append(fpath)
    return len(all_fpath), all_fpath
