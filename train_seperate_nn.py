import sys
import logging
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Nadam
from keras.optimizers import Adadelta
from keras.optimizers import SGD

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint


NB_EPOCH_MY_MODEL = 100



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

        y_pred = self.model.predict_proba(self.X_val, verbose=0)
        current = roc_auc_score(self.y_val, y_pred)
        print("\ninterval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, current))

        if current >= self.best:
            if self.verbose > 0:
                print('Epoch %05d: ROC AUC improved from %0.5f to %0.5f,'
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
                print('Epoch %05d: ROC AUC did not improve' %
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
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1


def my_model(input_dim):
    model = Sequential([
        GaussianNoise(sigma=0.0007, input_shape=input_dim),
        Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
        # BatchNormalization(axis=1),
        Activation('relu'),
        Convolution2D(16, 5, 5, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
        # BatchNormalization(axis=1),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.5),
        Convolution2D(32, 7, 7, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
        # BatchNormalization(axis=1),
        Activation('relu'),
        Convolution2D(32, 13, 13, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
        # BatchNormalization(axis=1),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        Dropout(0.5),
        Flatten(),
        Dense(200, init='he_normal', W_regularizer=l1(0.0001)),
        PReLU(init='zero', weights=None),
        Dropout(0.5),
        Dense(200, init='he_normal', W_regularizer=l2(0.0001)),
        PReLU(init='zero', weights=None),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid'),
    ])
    optimizer = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0005)
    # optimizer = SGD(lr=0.01, momentum=0.95)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  # metrics=['binary_crossentropy', my_roc_auc_score]
                  )
    return model


def train_my_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname_pre):
    input_dim = subtrain_x.shape[1:]
    print('input dim: {}'.format(input_dim))
    class_count = Counter(subtrain_y.tolist())
    class_weight = {}
    for c in class_count:
        class_weight[c] = len(subtrain_y) / class_count[c]
    print('class_weight: {}'.format(class_weight))
    model = my_model(input_dim)
    # best_model_fname = best_model_fname_pre + 'model.{epoch:02d}-{val_loss:.6f}.hdf5'
    best_model_fname = best_model_fname_pre + 'model.h5'

    # callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        verbose=1,
        factor=0.5,
        patience=4,
        min_lr=1e-8
    )
    early_stopping = EarlyStoppingByAUC(
        validation_data=(validation_x, validation_y),
        patience=15,
        verbose=1,
    )
    model_auc_cp = ModelAUCCheckpoint(
        best_model_fname,
        validation_data=(validation_x, validation_y),
        verbose=1,
        save_weights_only=False,
    )

    model.fit(
        subtrain_x,
        subtrain_y,
        nb_epoch=NB_EPOCH_MY_MODEL,
        batch_size=32,
        validation_data=(validation_x, validation_y),
        verbose=1,
        callbacks=[reduce_lr, early_stopping, model_auc_cp],
        # class_weight=class_weight,
    )

    model = load_model(best_model_fname)
    loss = model.evaluate(validation_x, validation_y, verbose=0)
    return model, loss


def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    data = joblib.load(data_fname)
    models = {}

    df_sub = pd.DataFrame({'File': [], 'Class': []})
    mean_auc = []
    for pid in sorted(data):
        print(pid)
        patient_data = data[pid]

        subtrain_x = patient_data['subtrain_x']
        subtrain_y = patient_data['subtrain_y']
        validation_x = patient_data['validation_x']
        validation_y = patient_data['validation_y']
        test_x = patient_data['test_x']
        test_fnames = patient_data['test_fnames']

        best_model_fname = '../' +pid + '_' + datetime.now().strftime("%Y%m%d%H") + '_'
        model, auc = train_my_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname)
        models[pid] = model

        test_pred = model.predict(test_x).flatten()
        mean_auc.append((auc, len(validation_y)))
        new_sub = pd.DataFrame({'File': test_fnames, 'Class': test_pred})
        df_sub = pd.concat([df_sub, new_sub], axis=0)

    weighted_auc = np.sum([auc*weight for auc, weight in mean_auc]) / np.sum([weight for _, weight in mean_auc])
    print(weighted_auc)
    fname = sub_fname.split('.')
    new_sub_fname = fname[0] + "{:.6f}".format(weighted_auc) + fname[1]
    df_sub.set_index('File').to_csv(new_sub_fname)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
