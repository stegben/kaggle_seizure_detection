import numpy as np
from joblib import Memory
from keras.callbacks import Callback

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


class ModelAUCCheckpoint(Callback):
    def __init__(self, filepath, validation_data, verbose=0, save_weights_only=False):
        super(ModelAUCCheckpoint, self).__init__()
        self.verbose = verbose
        self.filepath = filepath
        self.X_val, self.y_val = validation_data
        self.save_weights_only = save_weights_only
        self.best = 0.

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)

        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        current = roc_auc_score(self.y_val, y_pred)
        print("\ninterval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, current))

        if current >= self.best:
            if self.verbose > 0:
                print('\nEpoch %05d: ROC AUC improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch, self.best,
                         current, filepath))
            self.best = current
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: ROC AUC did not improve' %
                      (epoch))


class EarlyStoppingByAUC(Callback):
    def __init__(self, validation_data,  patience=0, verbose=0):
        super(EarlyStoppingByAUC, self).__init__()

        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.X_val, self.y_val = validation_data

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        current = roc_auc_score(self.y_val, y_pred)

        if current >= self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1
