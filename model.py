#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import join as pj

from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential


base_dir = os.path.dirname(__file__)
weights_dir = pj(base_dir, 'weights')


def get_model(model=1, input_size=16, output_size=16, maxlen=90):
    assert model in [1, 2]
    if model == 1:
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, input_size)))
        model.add(Dense(output_size))
        model.add(Activation('sigmoid'))

    elif model == 2:
        model = Sequential()
        model.add(LSTM(256, input_shape=(maxlen, input_size), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(256))
        model.add(Dropout(0.6))
        model.add(Dense(output_size))
        model.add(Activation('sigmoid'))

    return model


def get_weight_path(model, input_size, output_size, maxlen):
    template = "lstm_{}_layers_{}_input_size_{}_output_size_{}_maxlen.hdf5"
    return pj(weights_dir, template.format(model, input_size, output_size, maxlen))
