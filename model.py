#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential


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
