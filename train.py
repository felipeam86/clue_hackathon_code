#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script used to train the LSTM model
"""

import joblib
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam

from model import get_model, get_weight_path

# ====================== Default values ======================
INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 90
STEP_DAYS = 3
BATCH_SIZE = 256
NB_EPOCH = 15
MODEL = 1
N_TRAIN = 100000
N_TEST = 50000
# ============================================================


def reformat(df_in,
             input_size=INPUT_SIZE,
             output_size=OUTPUT_SIZE,
             maxlen=MAXLEN,
             step_days=STEP_DAYS,
             max_sequences=N_TRAIN):
    """Turns raw input into pairs of Xs and ys to train the network

    Xs correspond to sequences of maxlen days, and ys correspond to the maxlen+1 day

    Parameters
    ----------
    df_in: numpy.array
    input_size: int
        Number of input symptoms to take into account
    output_size: int
        Number of outputs symptoms to predict
    maxlen: int
        Number of days in the past that are required to generate next day prediction
    step_days: int
        Number of days to skip between two Xs
    max_sequences: int
        maximum number of sequences to generate for training. Ideally we would not limit it
        but due to memory constraints, pushing it way beyond 300000 is difficult on a 16Gb machine

    Returns
    -------
    days_sequence: np.array
        Tensor of 3 dimensions: user/sequence of days (maxlen)/symptoms (input_size)
        Xs: The symptom for the n first days for all users
    next_day: np.array
        Array of 2 dimensions: user/symptoms (output_size)
        ys: The symptoms for the n+1 day for all users
    """
    # Container for the Xs
    days_sequence = np.empty((max_sequences, maxlen, input_size), dtype=int)
    # Container for the ys
    next_day = np.empty((max_sequences, output_size), dtype=int)
    # The sequence of absolute days which we wil use to tell one user from another
    days = df_in.loc[:, "absolute_day"].reset_index(drop=True)
    df = df_in.reset_index(drop=True)

    # Counter for the number of sequences created
    j = 0
    # Last absolute day seen
    last_day = 0
    # Current absolute day being processed
    day_i = 0
    # Loop as long as we don't reach the end of the raw input
    #   or as long as we have created less than the max number of sequences
    while (day_i < (df.shape[0] - maxlen)) & (j < max_sequences):
        # Ensure that the beginning and end of the sequence correspond to the same user
        #   i.e. the last_day is anterior to current dat
        if last_day < days.ix[day_i + maxlen]:
            # Store the Xs
            days_sequence[j] = df.ix[day_i: day_i + maxlen - 1, :input_size]
            # Store the y
            next_day[j] = df.ix[day_i + maxlen, :output_size]
            # Increment sequence counter
            j += 1
        # move along the raw input by step_days
        day_i += step_days
        # Update counters
        last_day = days.ix[day_i]


    # In case less sequence than the max have been created, shorted the outputs
    days_sequence = days_sequence[:j, :, :]
    next_day = next_day[:j, :]

    print("Created %d sequences" % j)

    return days_sequence, next_day


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-N_train', default=N_TRAIN, type=int, help='Number of sequences used for training.')
    parser.add_argument('-N_test', default=N_TEST, type=int, help='Number of sequences used for testing.')
    parser.add_argument('-N_epochs', default=NB_EPOCH, type=int, help='Number of epochs')
    parser.add_argument('-batch_size', default=BATCH_SIZE, type=int, help='Batch size')
    parser.add_argument('-input_size', default=INPUT_SIZE, type=int, help='Input size')
    parser.add_argument('-output_size', default=OUTPUT_SIZE, type=int, help='Output size')
    parser.add_argument('-maxlen', default=MAXLEN, type=int, help='Max length of the sequence')
    parser.add_argument('-step_days', default=STEP_DAYS, type=int, help='STEP_DAYS')
    parser.add_argument('-model', default=MODEL, type=int, help="1 or 2 layers model", choices=[1, 2])
    parser.add_argument('-weights', default=None, type=str, help="Where to store the weights after training")
    parser.add_argument('-debug', action='store_true', help="If True, use a reduced subset of the data.")

    args = parser.parse_args()

    if args.weights is None:
        weights_backup = get_weight_path(args.model, args.input_size, args.output_size, args.maxlen)
    else:
        weights_backup = args.weights

    if args.debug:
        df_train = joblib.load('data/small_df_train.pkl.gz')
        df_test = joblib.load('data/small_df_test.pkl.gz')
    else:
        from preprocessing import get_features
        df_train, df_test = get_features(split=True)

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

    model = get_model(args.model, args.input_size, args.output_size, args.maxlen)
    model.compile(loss='binary_crossentropy', optimizer=adam(), metrics=['accuracy'])

    # Define callback to save model
    save_snapshots = ModelCheckpoint(weights_backup,
                                     monitor='val_loss',
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
