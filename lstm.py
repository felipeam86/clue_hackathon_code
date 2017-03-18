from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop, adam
from keras.utils.data_utils import get_file
from keras.callbacks import History, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prepare_data import transform_users, get_sample_of_users


INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 60
STEP_DAYS = 3
BATCH_SIZE = 512
NB_EPOCH = 3


def reformat(df_in,
             input_size=INPUT_SIZE,
             output_size=OUTPUT_SIZE,
             maxlen=MAXLEN,
             step_days=STEP_DAYS):

    max_sequences = 10 ** 5
    days_sequence = np.empty((max_sequences, maxlen, input_size), dtype=int)
    next_day = np.empty((max_sequences, output_size), dtype=int)
    days = df_in[:, -1]
    df = df_in[:, :-1]

    j = 0
    last_day = 0
    for day_i in range(0, df.shape[0] - maxlen, step_days):
        if last_day < days[day_i + maxlen]:
            days_sequence[j] = df[day_i: day_i + maxlen, :input_size]
        next_day[j] = df[day_i + maxlen, :output_size]
        j += 1
        last_day = days[day_i]

    days_sequence = days_sequence[:j, :, :]
    next_day = next_day[:j, :]

    return days_sequence, next_day


def plot_logs(history):
    """
    Plot the accuracy and loss for
        training and test sets
    """
    evaluation_cost = history.history['val_loss']
    evaluation_accuracy = history.history['val_acc']
    training_cost = history.history['loss']
    training_accuracy = history.history['acc']
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_figwidth(10)
    ax1.plot(evaluation_cost,label= 'test')
    ax1.plot(training_cost, label='train')
    ax1.set_title('Cost')
    ax1.legend()
    ax2.plot(evaluation_accuracy, label='test')
    ax2.plot(training_accuracy, label='train')
    ax2.set_title('Accuracy')
    ax2.legend(loc='lower right')


def sample(preds, temperature=1.0):
    """
    Generate the next sequence
    Low temperature means very conservative (picks more probable most of the time)
    High temperature means very adventurous (picks less probable more frequently)
    """
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_prediction(history,
                        model,
                        days=28,
                        maxlen=MAXLEN,
                        input_size=INPUT_SIZE,
                        output_size=OUTPUT_SIZE,
                        diversity=1):
    """
    Generates as many days of prediction as requested
    Considers maxlen days of past history (must be aligned with model)
    """
    generated = np.zeros((days,input_size))
    if history.shape[1]>maxlen:
        x = history[:,-61:-1,:input_size]
    else:
        x = history[:,:,:output_size]
    #print(x.shape)
    for i in range(days):
        #print("Day %d" % i)
        preds = model.predict(x, verbose=0)[0].reshape(output_size)
        print(preds.shape)
        #next_symptoms = sample(preds, diversity)
        next_symptoms = preds
        #print(next_symptoms)

        generated[i,:] = next_symptoms
        x[:,:maxlen-1,:] = x[:,1:,:]
        x[:,maxlen-1,:] = next_symptoms

    return generated


def get_model(model=1):
    assert model in [1, 2]
    if model == 1:
        filepath = "lstm_1_layer.hdf5"
        model = Sequential()
        model.add(LSTM(128, input_shape=(MAXLEN, INPUT_SIZE)))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('sigmoid'))

    elif model == 2:
        filepath = "lstm_2_layers.hdf5"
        model = Sequential()
        model.add(LSTM(256, input_shape=(MAXLEN, INPUT_SIZE), return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(256))
        model.add(Dropout(0.1))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=adam(),
                  metrics=['accuracy'])

    return model, filepath


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-N_train', default=100, type=int,
        help='Number of users for training dataset')

    parser.add_argument(
        '-N_test', default=10, type=int,
        help='Number of users for testing raining dataset')

    args = parser.parse_args()

    sample_of_users = get_sample_of_users(args.N_train + args.N_test)

    df_train = transform_users(sample_of_users[:args.N_train])
    df_test = transform_users(sample_of_users[-args.N_test:])
