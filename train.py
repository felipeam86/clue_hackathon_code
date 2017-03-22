#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import join as pj

import joblib
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam

from model import get_model

base_dir = os.path.dirname(__file__)
weights_dir = pj(base_dir, 'weights')

# ====================== Default values ======================
INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 90
STEP_DAYS = 3
BATCH_SIZE = 256
NB_EPOCH = 15
MODEL = 1
WEIGHTS_1 = pj(weights_dir, "lstm_1_layer.hdf5")
WEIGHTS_2 = pj(weights_dir, "lstm_2_layers_higher_dropout.hdf5")
N_TRAIN = 100000
N_TEST = 50000
# ============================================================


def reformat(df_in,
             input_size=INPUT_SIZE,
             output_size=OUTPUT_SIZE,
             maxlen=MAXLEN,
             step_days=STEP_DAYS,
             max_sequences=N_TRAIN):
    days_sequence = np.empty((max_sequences, maxlen, input_size), dtype=int)
    next_day = np.empty((max_sequences, output_size), dtype=int)
    days = df_in.loc[:, "absolute_day"].reset_index(drop=True)
    df = df_in.reset_index(drop=True)

    j = 0
    last_day = 0
    day_i = 0
    while (day_i < (df.shape[0] - maxlen)) & (j < max_sequences):
        if last_day < days.ix[day_i + maxlen]:
            days_sequence[j] = df.ix[day_i: day_i + maxlen - 1, :input_size]
        next_day[j] = df.ix[day_i + maxlen, :output_size]
        j += 1
        day_i += step_days
        last_day = days.ix[day_i]

    days_sequence = days_sequence[:j, :, :]
    next_day = next_day[:j, :]

    print("Created %d sequences" % j)

    return days_sequence, next_day


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-N_train', default=N_TRAIN, type=int, help='Number of users for training dataset')
    parser.add_argument('-N_test', default=N_TEST, type=int, help='Number of users for testing raining dataset')
    parser.add_argument('-N_epochs', default=NB_EPOCH, type=int, help='Number of epochs')
    parser.add_argument('-batch_size', default=BATCH_SIZE, type=int, help='Batch size')
    parser.add_argument('-input_size', default=INPUT_SIZE, type=int, help='Input size')
    parser.add_argument('-output_size', default=OUTPUT_SIZE, type=int, help='Output size')
    parser.add_argument('-maxlen', default=MAXLEN, type=int, help='maxlen')
    parser.add_argument('-step_days', default=STEP_DAYS, type=int, help='STEP_DAYS')
    parser.add_argument('-model', default=MODEL, type=int, help="1 or 2 layers model", choices=[1, 2])
    parser.add_argument('-weights', default=None, type=str, help="Where to store the weights after training")
    parser.add_argument('-debug', action='store_true', help="If True, use a reduced subset of the data.")

    args = parser.parse_args()

    if args.weights is None:
        if args.model == 1:
            weights_backup = WEIGHTS_1
        elif args.model == 2:
            weights_backup = WEIGHTS_2
    else:
        weights_backup = args.weights

    if args.debug:
        df_train = joblib.load('data/small_df_train.pkl.gz')
        df_test = joblib.load('data/small_df_test.pkl.gz')
    else:
        from preprocessing import get_training_data
        df_train, df_test = get_training_data(split=True)

    X_train, y_train = reformat(df_train,
                                input_size=args.input_size,
                                output_size=args.output_size,
                                maxlen=args.maxlen,
                                step_days=args.step_days,
                                max_sequences=args.N_train)
    del df_train
    X_test, y_test = reformat(df_test,
                              input_size=args.input_size,
                              output_size=args.output_size,
                              maxlen=args.maxlen,
                              step_days=args.step_days,
                              max_sequences=args.N_test)
    del df_test

    model = get_model(args.model)
    model.compile(loss='binary_crossentropy', optimizer=adam(), metrics=['accuracy'])

    # Define callback to save model
    save_snapshots = ModelCheckpoint(weights_backup,
                                     monitor='loss',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     verbose=0)

    train_history = model.fit(X_train,
                              y_train,
                              batch_size=args.batch_size,
                              nb_epoch=args.N_epochs,
                              validation_data=(X_test, y_test),
                              callbacks=[save_snapshots],
                              verbose=1)

    score = model.evaluate(X_test, y_test, verbose=2)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
