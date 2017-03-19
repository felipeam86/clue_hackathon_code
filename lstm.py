from __future__ import print_function

try:
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import adam

from prepare_data import transform_users, transform_user, get_sample_of_users, symptoms_of_interest_dict, cycles0

INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 60
STEP_DAYS = 3
BATCH_SIZE = 256
NB_EPOCH = 15
MODEL = 1
WEIGHTS = "lstm_1_layer.hdf5"

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
    return f


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
                        output_size=OUTPUT_SIZE):
    """
    Generates as many days of prediction as requested
    Considers maxlen days of past history (must be aligned with model)
    """
    generated = np.zeros((days, output_size))
    if history.shape[1] > maxlen:
        x = history[:, -maxlen - 1:-1, :input_size]
    else:
        x = history[:, :, :input_size]
    for i in range(days):
        preds = model.predict(x, verbose=0)[0].reshape(output_size)
        generated[i, :] = preds

        if input_size > output_size:
            res = np.zeros(input_size)
            res[:output_size] = preds
            preds = res

        next_symptoms = preds
        x[:, :maxlen - 1, :] = x[:, 1:, :]
        x[:, maxlen - 1, :] = next_symptoms

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
        filepath = "lstm_2_layers_higher_dropout.hdf5"
        model = Sequential()
        model.add(LSTM(256, input_shape=(MAXLEN, INPUT_SIZE), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(256))
        model.add(Dropout(0.6))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('sigmoid'))

    return model, filepath


def format_prediction(prediction, user_id):
    output = []
    prediction = pd.DataFrame(prediction)
    for i, row in prediction.iterrows():
        for j, symptom in enumerate(row):
            line = [user_id, i + 1, j, prediction.ix[i, j]]
            output.append(line)
    return output


def pad_reshape_history(sequence, maxlen, input_size):
    if sequence.shape[0] < maxlen:
        hist = np.zeros((maxlen, sequence.shape[1]))
        hist[maxlen - sequence.shape[0]:, :] = sequence
    else:
        hist = sequence[-maxlen - 1:-1, :]
    if sequence.shape[1] > input_size:
        hist = hist[:, :input_size]
    hist = hist.reshape(1, maxlen, -1)
    return hist


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-N_train', default=100, type=int,
        help='Number of users for training dataset')

    parser.add_argument(
        '-N_test', default=10, type=int,
        help='Number of users for testing raining dataset')

    parser.add_argument(
        '-N_epochs', default=NB_EPOCH, type=int,
        help='Number of epochs')

    parser.add_argument(
        '-batch_size', default=BATCH_SIZE, type=int,
        help='Batch size')

    parser.add_argument(
        '-input_size', default=INPUT_SIZE, type=int,
        help='Input size')

    parser.add_argument(
        '-output_size', default=OUTPUT_SIZE, type=int,
        help='Output size')

    parser.add_argument(
        '-maxlen', default=MAXLEN, type=int,
        help='maxlen')

    parser.add_argument(
        '-step_days', default=STEP_DAYS, type=int,
        help='STEP_DAYS')

    parser.add_argument('-fit', action='store_true',
                        help="If True, fit the model.")

    parser.add_argument(
        '-model', default=MODEL, type=int,
                        help="1 = 1 layer, 2 = 2 layers")

    parser.add_argument(
        '-weights', default=WEIGHTS, type=str,
                        help="Use the submitted pre-trained model")

    args = parser.parse_args()

    INPUT_SIZE = args.input_size
    OUTPUT_SIZE = args.output_size
    MAXLEN = args.maxlen
    STEP_DAYS = args.step_days
    MODEL = args.model
    WEIGHTS = args.weights

    sample_of_users = get_sample_of_users(args.N_train + args.N_test)

    df_train = transform_users(sample_of_users[:args.N_train])
    df_test = transform_users(sample_of_users[-args.N_test:])

    model, filepath = get_model(MODEL)

    # Define callback to save model
    save_snapshots = ModelCheckpoint(filepath,
                                     monitor='loss',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     verbose=0)

    X_train, y_train = reformat(df_train)
    X_test, y_test = reformat(df_test)

    if args.fit:
        model.compile(loss='binary_crossentropy', optimizer=adam(), metrics=['accuracy'])
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
    else:
        # load weights
        model.load_weights(WEIGHTS)
        # Compile model (required to make predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Created model and loaded weights from file")

    submission = []
    j = 0
    for index, woman in cycles0.iterrows():
        current_id = woman.user_id
        expected_length = int(np.ceil(woman.expected_cycle_length))
        sequence = transform_user(current_id)
        hist = pad_reshape_history(sequence, MAXLEN, INPUT_SIZE)
        res = generate_prediction(hist, model, maxlen=MAXLEN, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE,
                                  days=expected_length)
        submission.append(format_prediction(res, current_id))

    submission_df = pd.concat([pd.DataFrame(submission[i]) for i in range(len(submission))], ignore_index=True)
    submission_df.columns = ['user_id', 'day_in_cycle', 'symptom', 'probability']
    submission_df["symptom"] = submission_df["symptom"].apply(lambda x: symptoms_of_interest_dict[x])

    submission_df.to_csv("./result.txt", index=False)