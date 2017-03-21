from os.path import join as pj
import os
import joblib
import pandas as pd
import numpy as np

base_dir = os.path.dirname(__file__)
data_dir = pj(base_dir, 'data')

# ====================== Import data ======================
active_days = pd.read_csv(pj(data_dir, 'active_days.csv'), parse_dates=['date'])
users = pd.read_csv(pj(data_dir, 'users.csv'))

cycles = pd.read_csv(pj(data_dir, 'cycles.csv'), parse_dates=['cycle_start'])
cycles0 = pd.read_csv(pj(data_dir, 'cycles0.csv'), parse_dates=['cycle_start'])

# Train on this
tracking = pd.read_csv(pj(data_dir, 'tracking.csv'), parse_dates=['date'])
# Test on this
tracking_test = pd.read_csv(pj(data_dir, 'labels.csv'))


# ================  Feature engineer a user ================
symptoms_of_interest = [
    'happy', 'pms', 'sad', 'sensitive_emotion',
    'energized', 'exhausted', 'high_energy', 'low_energy',
    'cramps', 'headache', 'ovulation_pain', 'tender_breasts',
    'acne_skin', 'good_skin', 'oily_skin', 'dry_skin'
]

other_symptoms = list(set(tracking.symptom.unique()) - set(symptoms_of_interest))
list_of_symptoms = symptoms_of_interest + other_symptoms
ordered_symptoms = {s: i for i, s in enumerate(list_of_symptoms)}
symptoms_of_interest_dict = {code:symptom for symptom, code in ordered_symptoms.items() if code < 16}
N_symptoms = len(ordered_symptoms)
training_columns = list_of_symptoms + ['day_in_cycle', 'absolute_day', 'period']


def expand_cycle(cycle):
    dates = pd.date_range(start=cycle.cycle_start, periods=cycle.cycle_length).tolist()
    period = np.zeros(cycle.cycle_length, dtype=np.int8)
    period[:cycle.period_length] = 1
    day_in_cycle = np.arange(1, cycle.cycle_length + 1, dtype=np.int8)

    index = pd.MultiIndex.from_tuples(
        tuples=list(zip([cycle.user_id] * int(cycle.cycle_length), dates)),
        # [(cycle.user_id, cycle.cycle_id)] * int(cycle.cycle_length),
        names=["user_id", "date"]
    )

    return pd.DataFrame(
        data=list(zip([cycle.cycle_id] * int(cycle.cycle_length), day_in_cycle, period)),
        index=index,
        columns=['cycle_id', 'day_in_cycle', 'period']
    )


def expand_cycles(cycles):
    try:
        cycles_processed = joblib.load("../data/cycles_processed.pkl.gz")
    except:
        cycles_processed = pd.concat([expand_cycle(cycle) for _, cycle in cycles.iterrows()])
        joblib.dump(cycles_processed, "../data/cycles_processed.pkl.gz")

    return cycles_processed


def process_tracking(tracking):
    # One hot encode the symptoms
    tracking_processed = pd.get_dummies(
        tracking[["user_id", "date", "symptom"]],
        columns=['symptom'], prefix='', prefix_sep=''
    )

    # Aggregate symptoms per day
    return tracking_processed.groupby(['user_id', 'date']).sum()[list_of_symptoms]


def get_training_data(split=True, force=False):

    training_backup = "../data/training.pkl.gz"

    # Try to load from memory if already computed
    if os.path.exists(training_backup) and not force:
        training = joblib.load(training_backup)
    else:
        # Expand cycles so that there is a line per date (active or not) with a boolean indicator of period
        cycles_processed = expand_cycles(cycles)

        # Expand tracking so that there is a line per date (active or not) with a one hot encoded symtoms
        tracking_processed = process_tracking(tracking)

        # Merge cycles and tracking information
        training = pd.merge(cycles_processed, tracking_processed, left_index=True, right_index=True, how='outer').fillna(0)

        # Find the first day the user started using the app
        training = pd.merge(
            training,
            cycles.groupby('user_id')\
                  .agg({'cycle_start': {'first_use': 'min'}})\
                  .reset_index()\
                  .set_index('user_id')['cycle_start'],
            left_index=True,
            right_index=True
        )

        # Find the absolute day for each row from the day the user started using the app
        absolute_day = (training.reset_index().date.dt.date - training.reset_index().first_use.dt.date).dt.days + 1
        absolute_day.index = training.index
        training['absolute_day'] = absolute_day
        # Keep only the columns needed by the RNN
        training = training[training_columns]

        # Make a copy to speed up development iterations
        joblib.dump(training, training_backup)

        # This saves memory, I think...
        del tracking_processed
        del cycles_processed

    if split:
        # Do a train/test split of the data
        train_users = users.user_id.sample(frac=0.8)
        training = training.reset_index()
        df_train = training[training.user_id.isin(train_users)]
        df_test = training[~training.user_id.isin(train_users)]
        return df_train, df_test
    else:
        return training


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-N_users', default=10, type=int,
        help='How many users to sequence')

    args = parser.parse_args()
