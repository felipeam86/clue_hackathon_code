#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script used to predict with an already trained LSTM model
"""

from os.path import join as pj

import numpy as np
import pandas as pd

from model import get_model, get_weight_path
from preprocessing import symptoms_of_interest_dict, data_dir, prepare_data_for_prediction

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
    """Generates as many days of prediction as requested

    Parameters
    ----------
    history: np.array
        Tensor of 3 dimensions: user/sequence of days (maxlen)/symptoms
        History of m days (as per "maxlen" parameter) of symptoms for users
    model: keras.model
    days: int
    maxlen: int
        Number of days in the past that are required to generate next day prediction
    input_size: int
        Number of input symptoms to take into account
    output_size: int
        Number of outputs symptoms to predict

    Returns
    -------
    generated: np.array
        Tensor of 3 dimensions: user/sequence of days/symptoms
        Predicted n days (as per "days" parameter) of symptoms of all users in input
    """
    print("Generating for expected cycle length %d" % days)

    # Create container for the output
    generated = np.zeros((history.shape[0], days, output_size))

    # Create a copy of input to avoid modifying original data
    x = history.copy()

    # If input has more than the 16 symptoms to predict
    # create a container res that will be used to pad the missing data
    # before submitting for next days predictions
    if input_size > output_size:
        res = np.zeros((history.shape[0], 1, input_size))
        non_symptoms = np.zeros((history.shape[0], 1, 3))
        non_symptoms[:, :, 0:2] = history[:, -1:, 81:83] + 1
        non_symptoms[:, :, 2] = 1
    # non_symptoms[:,:,3:5] = history[:,history.shape[1],-2:0]

    # Generate predictions for each days in maxlen
    for i in range(days):
        preds = model.predict(x, verbose=0)[0].reshape(output_size)
        generated[:, i, :] = preds

        # If input has more than the 16 symptoms to predict
        # pad the missing data
        if input_size > output_size:
            res[:, :, :output_size] = preds
            res[:, :, -3:] = non_symptoms
            non_symptoms[:, :, 0:2] += 1
            if i > 3:
                non_symptoms[:, :, 2] = 0
            preds = res

        # next_symptoms = sample(preds, diversity)
        next_symptoms = preds

        # remove the first day of history
        x[:, :maxlen - 1, :] = x[:, 1:, :]
        # add the predicting data as last day of history
        x[:, -1:, :] = next_symptoms

    return generated


def get_submission(model, sequence, cycles_predict,
                   input_size=INPUT_SIZE,
                   output_size=OUTPUT_SIZE,
                   maxlen=MAXLEN):
    """Split user by expected cycle length, and predict/format symptoms for corresponding number of days

    Parameters
    ----------
    model: keras.model
    sequence: np.array
        Tensor of 3 dimensions: user/sequence of days (maxlen)/symptoms
        History of m days (as per "maxlen" parameter) of symptoms for all users
    cycles_predict: np.array
        Contains expected cycle length for each user
    input_size: int
    output_size: int
    maxlen: int

    Returns
    -------
    submission: np.array
        Tensor of 3 dimensions: user/sequence of days/symptoms
        Predicted n days (as per "days" parameter) of symptoms of all users
    """
    # Create empty data frame that will contain result
    submission = pd.DataFrame(columns=['user_id', 'day_in_cycle', 'symptom', 'probability'])

    # Create a list of all unique expected cycles length
    expected_lengths = list(set(cycles_predict.expected_cycle_length))

    # Loop through each unique expected cycle length and predict/format
    for expected_length in expected_lengths:
        # Create user_id list of women with considered cycle length
        women = list(cycles_predict[cycles_predict.expected_cycle_length == expected_length].user_id)
        # Create symtoms subset for all considered women and reshape it
        women_to_predict = sequence.loc[women]
        women_to_predict = women_to_predict.ix[:, :input_size]
        women_to_predict = np.array(women_to_predict).reshape(-1, maxlen, input_size)
        expected_length = int(np.ceil(expected_length))
        # Generate symptoms predictions
        res = generate_prediction(women_to_predict, model, maxlen=maxlen, input_size=input_size,
                                  output_size=output_size, days=expected_length)
        # Reshape symtoms predictions in the format expected by statice
        formatted_submission = format_prediction(res.reshape(-1, 16), output_size, women, expected_length)
        # Concatenate result for this cycle length with previous outputs for other cycle lengths
        submission = pd.concat([submission, formatted_submission])

    return submission


def format_prediction(prediction,
                      output_size,
                      women,
                      expected_cycle_length):
    """Convert prediction to a format expected by the statice plateform

    Parameters
    ----------
    prediction: np.array
        Tensor of 3 dimensions: user/sequence of days (maxlen)/symptoms
        Prediction of m days (as per "expected_cycle_length" parameter) of symptoms for all users
    output_size: int
    women: list
        List of user ids for which the prediction is provided
    expected_cycle_length: int

    Returns
    -------
    output: np.array
        Array of 2 dimensions: symptom per woman/probability
    """
    print("Formatting for expected cycle length %d" % expected_cycle_length)
    # Create column with user_id
    user = pd.Series(list(np.repeat(women, expected_cycle_length)) * output_size, name="user_id")
    # Create column with day_in_cycle
    day_in_cycle = list(range(1, expected_cycle_length + 1)) * (len(women) * output_size)
    # Melt the dataframe to long format
    s = pd.melt(pd.DataFrame(prediction).reset_index(), id_vars="index")
    # Concatenate user_id and day_in_cycle
    s["index"] = day_in_cycle
    s.columns = ['day_in_cycle', 'symptom', 'probability']
    s["symptom"] = s["symptom"].apply(lambda x: symptoms_of_interest_dict[x])
    output = pd.concat([user, s], axis=1)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-input_size', default=INPUT_SIZE, type=int, help='Input size')
    parser.add_argument('-output_size', default=OUTPUT_SIZE, type=int, help='Output size')
    parser.add_argument('-maxlen', default=MAXLEN, type=int, help='maxlen')
    parser.add_argument('-model', default=MODEL, type=int, help="1 or 2 layers model", choices=[1, 2])
    parser.add_argument('-weights', default=None, type=str, help="Where to load pretrained weights")

    args = parser.parse_args()

    # Get weights name
    if args.weights is None:
        weights_backup = get_weight_path(args.model, args.input_size, args.output_size, args.maxlen)
    else:
        weights_backup = args.weights

    # Load model
    model = get_model(args.model, args.input_size, args.output_size, args.maxlen)

    # load weights
    model.load_weights(weights_backup)

    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    # Read user_id for which to make the prediction
    cycles_predict = pd.read_csv(pj(data_dir, 'cycles0.csv'), parse_dates=['cycle_start'])

    # Format input data so it can be read by the neural network
    X_predict = prepare_data_for_prediction(maxlen=args.maxlen)
    # Predict symptoms and format them
    submission_df = get_submission(model, X_predict, cycles_predict,
                                   args.input_size, args.output_size, args.maxlen)
    # Write output to file
    submission_df.to_csv("./result.txt", index=False)
