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

from prepare_data_fast import symptoms_of_interest_dict, cycles0, load_sequence, get_sample_of_users, transform_users

INPUT_SIZE = 16
OUTPUT_SIZE = 16
MAXLEN = 90
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
    generated = np.zeros((history.shape[0],days,output_size))
    x = history
    for i in range(days):
        #print("Day %d" % i)
        preds = model.predict(x, verbose=0)[0].reshape(output_size)
        #print(preds)
        generated[:,i,:] = preds
        
        if input_size > output_size:
            res = np.zeros(input_size)
            res[:output_size] = preds
            preds = res

        #print(preds.shape)
        #next_symptoms = sample(preds, diversity)
        next_symptoms = preds
        #print(next_symptoms)


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
        filepath = "lstm_2_layers_higher_dropout.hdf5"
        model = Sequential()
        model.add(LSTM(256, input_shape=(MAXLEN, INPUT_SIZE), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(256))
        model.add(Dropout(0.6))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('sigmoid'))

    return model, filepath

def get_submission():
    submission = pd.DataFrame(columns=['user_id','day_in_cycle','symptom','probability'])
    
    expected_lengths = list(set(cycles0.expected_cycle_length))
    
    for expected_length in expected_lengths:
        women = list(cycles0[cycles0.expected_cycle_length == expected_length].user_id)
        women_to_predict = sequence.loc[women]
        women_to_predict = women_to_predict.ix[:,:INPUT_SIZE]
        #print(women_to_predict.shape)
        women_to_predict = np.array(women_to_predict).reshape(-1,90,INPUT_SIZE)
        expected_length = int(np.ceil(expected_length))

        res = generate_prediction(women_to_predict,model, maxlen=MAXLEN,input_size=INPUT_SIZE,output_size=OUTPUT_SIZE,days=expected_length)
        print(res.shape)
        formatted_submission = format_prediction(res.reshape(-1,16),OUTPUT_SIZE,women,expected_length)
        submission = pd.concat([submission,formatted_submission])

    return submission

def format_prediction(  prediction,
                        output_size,
                        women,
                        expected_cycle_length):

    expected_lengths = expected_cycle_length * output_size
    print("Formatting for expected cycle length %d" % (expected_lengths/output_size))
    user = pd.Series(np.repeat(women,expected_lengths),name="user_id")
    day_in_cycle = list(range(1,expected_cycle_length+1)) * (len(women)*output_size)
    
    s = pd.melt(pd.DataFrame(prediction).reset_index(),id_vars="index")
    s["index"] = day_in_cycle
    s.columns = ['day_in_cycle','symptom','probability']
    s["symptom"] = s["symptom"].apply(lambda x: symptoms_of_interest_dict[x])    
    
    output = pd.concat([user,s],axis=1)
    return output



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

    sequence = load_sequence()
    submission_df = get_submission()

    submission_df.to_csv("./result.csv", index=False)
