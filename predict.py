#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pj

import numpy as np
import pandas as pd

from model import get_model
from preprocessing import symptoms_of_interest_dict, data_dir, prepare_data_for_prediction
from train import WEIGHTS_1, WEIGHTS_2

# ====================== Default values ======================
INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 90
MODEL = 1
# ============================================================


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
    print("Generating for expected cycle length %d" % days)

    generated = np.zeros((history.shape[0], days, output_size))
    x = history

    if input_size > output_size:
        res = np.zeros((history.shape[0],1,input_size))
        non_symptoms = np.zeros((history.shape[0], 1, 3))
        non_symptoms[:, :, 0:2] = history[:, -1:, 81:83] + 1
        non_symptoms[:, :, 2] = 1
        # non_symptoms[:,:,3:5] = history[:,history.shape[1],-2:0]

    for i in range(days):
        preds = model.predict(x, verbose=0)[0].reshape(output_size)
        generated[:, i, :] = preds

        if input_size > output_size:
            res[:,:,:output_size] = preds
            res[:,:,-3:] = non_symptoms
            non_symptoms[:, :, 0:2] += 1
            if i > 3:
                non_symptoms[:, :, 2] = 0
            preds = res

        # next_symptoms = sample(preds, diversity)
        next_symptoms = preds

        x[:, :maxlen -1, :] = x[:, 1:, :]
        x[:, -1: , :] = next_symptoms

    return generated


def get_submission(model, sequence, cycles_predict,
                   input_size=INPUT_SIZE,
                   output_size=OUTPUT_SIZE,
                   maxlen=MAXLEN):
    submission = pd.DataFrame(columns=['user_id', 'day_in_cycle', 'symptom', 'probability'])

    expected_lengths = list(set(cycles_predict.expected_cycle_length))

    for expected_length in expected_lengths:
        women = list(cycles_predict[cycles_predict.expected_cycle_length == expected_length].user_id)
        women_to_predict = sequence.loc[women]
        women_to_predict = women_to_predict.ix[:, :input_size]
        women_to_predict = np.array(women_to_predict).reshape(-1, maxlen, input_size)
        expected_length = int(np.ceil(expected_length))

        res = generate_prediction(women_to_predict, model, maxlen=maxlen, input_size=input_size,
                                  output_size=output_size, days=expected_length)

        formatted_submission = format_prediction(res.reshape(-1, 16), output_size, women, expected_length)
        submission = pd.concat([submission, formatted_submission])

    return submission


def format_prediction(prediction,
                      output_size,
                      women,
                      expected_cycle_length):
    print("Formatting for expected cycle length %d" % expected_cycle_length)
    user = pd.Series(list(np.repeat(women, expected_cycle_length)) * output_size, name="user_id")
    day_in_cycle = list(range(1, expected_cycle_length + 1)) * (len(women) * output_size)

    s = pd.melt(pd.DataFrame(prediction).reset_index(), id_vars="index")
    s["index"] = day_in_cycle
    s.columns = ['day_in_cycle', 'symptom', 'probability']
    s["symptom"] = s["symptom"].apply(lambda x: symptoms_of_interest_dict[x])

    output = pd.concat([user, s], axis=1)
    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-input_size', default=INPUT_SIZE, type=int, help='Input size')
    parser.add_argument('-output_size', default=OUTPUT_SIZE, type=int, help='Output size')
    parser.add_argument('-maxlen', default=MAXLEN, type=int, help='maxlen')
    parser.add_argument('-model', default=MODEL, type=int, help="1 or 2 layers model", choices=[1, 2])
    parser.add_argument('-weights', default=None, type=str, help="Where to store the weights after training")

    args = parser.parse_args()

    if args.weights is None:
        if args.model == 1:
            weights_backup = WEIGHTS_1
        elif args.model == 2:
            weights_backup = WEIGHTS_2
    else:
        weights_backup = args.weights

    model = get_model(args.model, args.input_size, args.output_size, args.maxlen)

    # load weights
    model.load_weights(weights_backup)

    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    cycles_predict = pd.read_csv(pj(data_dir, 'cycles0.csv'), parse_dates=['cycle_start'])

    X_predict = prepare_data_for_prediction(maxlen=args.maxlen)
    submission_df = get_submission(model, X_predict, cycles_predict,
                                   args.input_size, args.output_size, args.maxlen)

    submission_df.to_csv("./result.txt", index=False)
