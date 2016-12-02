import sys
import uuid
import logging
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Input
from keras.layers import merge, Permute, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
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


def my_model(input_dim):
    image_input = Input(shape=input_dim)
    encoded_image = GaussianNoise(sigma=0.0015)(image_input)
    encoded_image = Convolution2D(16, 2, 2, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = MaxPooling2D(pool_size=(1, 2), dim_ordering='th')(encoded_image)
    encoded_image = Dropout(0.2)(encoded_image)
    encoded_image = Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = Convolution2D(16, 3, 4, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = MaxPooling2D(pool_size=(1, 2), dim_ordering='th')(encoded_image)
    encoded_image = Dropout(0.2)(encoded_image)
    encoded_image = Convolution2D(32, 3, 5, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = Convolution2D(32, 5, 5, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = MaxPooling2D(pool_size=(2, 2), dim_ordering='th')(encoded_image)
    encoded_image = Dropout(0.2)(encoded_image)

    encoded_image = Permute((3, 1, 2))(encoded_image)
    encoded_image = Reshape((9, 13*32))(encoded_image)

    encoded_image = LSTM(64, return_sequences=True, name='rec1', activation='tanh')(encoded_image)
    encoded_image = Dropout(0.1)(encoded_image)
    encoded_image = LSTM(128, return_sequences=False, name='rec2', activation='tanh')(encoded_image)
    encoded_image = Dropout(0.1)(encoded_image)
    # encoded_image = Flatten()(encoded_image)

    pid_input = Input(shape=(3,))
    encoded_pid = Dense(2, init='he_normal',W_regularizer=l2(0.2))(pid_input)
    encoded_pid = Activation('relu')(encoded_pid)
    encoded_pid = Dropout(0.3)(encoded_pid)

    # image_model = Sequential([
    #     GaussianNoise(sigma=0.0007, input_shape=input_dim),
    #     Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
    #     # BatchNormalization(axis=1),
    #     Activation('relu'),
    #     Convolution2D(16, 5, 5, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
    #     # BatchNormalization(axis=1),
    #     Activation('relu'),
    #     MaxPooling2D(pool_size=(3, 3)),
    #     Dropout(0.5),
    #     Convolution2D(32, 7, 7, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
    #     # BatchNormalization(axis=1),
    #     Activation('relu'),
    #     Convolution2D(32, 13, 13, border_mode='valid', dim_ordering='th', W_regularizer=l2(0.0001)),
    #     # BatchNormalization(axis=1),
    #     Activation('relu'),
    #     MaxPooling2D(pool_size=(5, 5)),
    #     Dropout(0.5),
    #     Flatten(),
    # ])
    #
    # pid_model = Sequential([
    #     Dense(20, init='he_normal', input_shape=(3,)),
    #     Activation('tanh'),
    #     Dropout(0.2),
    # ])
    #
    # encoded_pid = pid_model(pid_input)
    # encoded_image = image_model(image_model)
    merged = merge([encoded_image, encoded_pid], mode='concat')

    output = Dense(200, init='he_normal', W_regularizer=l2(0.000001))(merged)
    output = PReLU(init='zero', weights=None)(output)
    output = Dropout(0.3)(output)
    output = Dense(200, init='he_normal')(output)
    output = PReLU(init='zero', weights=None)(output)
    output = Dropout(0.3)(output)
    output = Dense(1)(output)
    output = Activation('sigmoid')(output)

    model = Model(input=[image_input, pid_input], output=output)

    optimizer = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0)
    # optimizer = SGD(lr=0.0002, momentum=0.95)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  # metrics=['binary_crossentropy', my_roc_auc_score]
                  )
    print(model.summary())
    return model


def train_my_model(subtrain_x, subtrain_pid, subtrain_y, validation_x, validation_pid, validation_y, best_model_fname_pre):
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
        cooldown=3,
        min_lr=1e-8
    )
    early_stopping = EarlyStoppingByAUC(
        validation_data=([validation_x, validation_pid], validation_y),
        patience=15,
        verbose=1,
    )
    model_auc_cp = ModelAUCCheckpoint(
        best_model_fname,
        validation_data=([validation_x, validation_pid], validation_y),
        verbose=1,
        save_weights_only=False,
    )

    model.fit(
        [subtrain_x, subtrain_pid],
        subtrain_y,
        nb_epoch=NB_EPOCH_MY_MODEL,
        batch_size=32,
        validation_data=([validation_x, validation_pid], validation_y),
        verbose=1,
        callbacks=[reduce_lr, early_stopping, model_auc_cp],
        # class_weight=class_weight,
    )

    model = load_model(best_model_fname)
    y_pred = model.predict([validation_x, validation_pid], verbose=0).flatten()
    loss = roc_auc_score(validation_y, y_pred)
    return model, loss


def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    data = joblib.load(data_fname)
    models = {}

    df_sub = pd.DataFrame({'File': [], 'Class': []})
    mean_auc = []
    subtrain_pid = []
    all_subtrain_x = []
    all_subtrain_y = []
    validation_pid = []
    all_validation_x = []
    all_validation_y = []
    test_pid = []
    test_fnames = []
    all_test_x = []
    for pid in sorted(data):
        print(pid)
        patient_data = data[pid]

        subtrain_x = patient_data['subtrain_x']
        subtrain_y = patient_data['subtrain_y']
        all_subtrain_x.append(subtrain_x)
        all_subtrain_y.append(subtrain_y)
        subtrain_pid = subtrain_pid + [pid for _ in range(len(subtrain_y))]

        validation_x = patient_data['validation_x']
        validation_y = patient_data['validation_y']
        all_validation_x.append(validation_x)
        all_validation_y.append(validation_y)
        validation_pid = validation_pid + [pid for _ in range(len(validation_y))]

        test_x = patient_data['test_x']
        test_fnames = test_fnames + patient_data['test_fnames']
        all_test_x.append(test_x)
        test_pid = test_pid + [pid for _ in range(test_x.shape[0])]

    subtrain_x = np.concatenate(all_subtrain_x)
    subtrain_pid = pd.get_dummies(subtrain_pid).sort_index(axis=1).values
    subtrain_y = np.concatenate(all_subtrain_y)
    validation_x = np.concatenate(all_validation_x)
    validation_y = np.concatenate(all_validation_y)
    validation_pid = pd.get_dummies(validation_pid).sort_index(axis=1).values

    best_model_fname = '../' + datetime.now().strftime("%Y%m%d%H") + '_' + str(uuid.uuid4()) + '_'
    model, auc = train_my_model(
        subtrain_x,
        subtrain_pid,
        subtrain_y,
        validation_x,
        validation_pid,
        validation_y,
        best_model_fname
    )
    print(auc)

    test_x = np.concatenate(all_test_x)
    test_pid = pd.get_dummies(test_pid).sort_index(axis=1).values
    test_pred = model.predict([test_x, test_pid]).flatten()
    mean_auc.append((auc, len(validation_y)))
    new_sub = pd.DataFrame({'File': test_fnames, 'Class': test_pred})
    df_sub = pd.concat([df_sub, new_sub], axis=0)
    df_sub['File'] = 'new_' + df_sub['File']
    fname = sub_fname.split('.')
    new_sub_fname = fname[0] + "{:.6f}".format(auc) + '.' + fname[1]
    df_sub.set_index('File').to_csv(new_sub_fname)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
