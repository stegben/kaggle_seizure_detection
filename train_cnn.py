import sys
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
from keras.layers import merge
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
from keras.callbacks import ReduceLROnPlateau

from utils import EarlyStoppingByAUC
from utils import ModelAUCCheckpoint


NB_EPOCH_MY_MODEL = 100


def my_model(input_dim):
    image_input = Input(shape=input_dim)
    encoded_image = GaussianNoise(sigma=0.0003)(image_input)
    encoded_image = Convolution2D(16, 2, 2, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = MaxPooling2D(pool_size=(1, 3), dim_ordering='th')(encoded_image)
    encoded_image = Dropout(0.3)(encoded_image)
    encoded_image = Convolution2D(32, 5, 5, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = Convolution2D(32, 7, 7, border_mode='valid', dim_ordering='th')(encoded_image)
    # encoded_image = BatchNormalization(axis=1)(encoded_image)
    encoded_image = Activation('relu')(encoded_image)
    encoded_image = MaxPooling2D(pool_size=(2, 10), dim_ordering='th')(encoded_image)
    encoded_image = Dropout(0.3)(encoded_image)
    encoded_image = Flatten()(encoded_image)

    pid_input = Input(shape=(3,))
    encoded_pid = Dense(3, init='he_normal',W_regularizer=l2(0.01))(pid_input)
    encoded_pid = Activation('relu')(encoded_pid)
    encoded_pid = Dropout(0.5)(encoded_pid)

    merged = merge([encoded_image, encoded_pid], mode='concat')

    output = Dense(200, init='he_normal', W_regularizer=l2(0.0001))(merged)
    output = PReLU(init='zero', weights=None)(output)
    output = Dropout(0.5)(output)
    output = Dense(200, init='he_normal')(output)
    output = PReLU(init='zero', weights=None)(output)
    output = Dropout(0.5)(output)
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

    best_model_fname = '../' + datetime.now().strftime("%Y%m%d%H") + '_'
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
