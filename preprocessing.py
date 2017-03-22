#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import join as pj

import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(__file__)
data_dir = pj(base_dir, 'data')

# ====================== Import data ======================
active_days = pd.read_csv(pj(data_dir, 'active_days.csv'), parse_dates=['date'])
users = pd.read_csv(pj(data_dir, 'users.csv'))

cycles = pd.read_csv(pj(data_dir, 'cycles.csv'), parse_dates=['cycle_start'])
cycles_predict = pd.read_csv(pj(data_dir, 'cycles0.csv'), parse_dates=['cycle_start'])

# Train on this
tracking = pd.read_csv(pj(data_dir, 'tracking.csv'), parse_dates=['date'])
# Test on this
tracking_test = pd.read_csv(pj(data_dir, 'labels.csv'))


# =============  Symptoms on the correct order ============
symptoms_of_interest = [
    'happy', 'pms', 'sad', 'sensitive_emotion',  # emotion
    'energized', 'exhausted', 'high_energy', 'low_energy',  # energy
    'cramps', 'headache', 'ovulation_pain', 'tender_breasts',  # pain
    'acne_skin', 'good_skin', 'oily_skin', 'dry_skin'  # skin
]

other_symptoms = [
    'fever_ailment', 'injury_ailment', 'cold_flu_ailment', 'allergy_ailment',  # ailment
    'vacation_appointment', 'doctor_appointment', 'date_appointment', 'ob_gyn_appointment',  # appointment
    'salty_craving', 'carbs_craving', 'sweet_craving', 'chocolate_craving',  # craving
    'bloated', 'nauseated', 'great_digestion', 'gassy',  # digestion
    'running', 'biking', 'yoga', 'swimming',  # exercise
    'atypical', 'egg_white', 'sticky', 'creamy',  # fluid
    'oily_hair', 'dry_hair', 'bad_hair', 'good_hair',  # hair
    'antibiotic_medication', 'cold_flu_medication', 'pain_medication', 'antihistamine_medication',  # medication
    'meditation',  # meditation
    'focused', 'calm', 'stressed', 'distracted',  # mental
    'motivated', 'unproductive', 'unmotivated', 'productive',  # motivation
    'hangover', 'cigarettes', 'big_night_party', 'drinks_party',  # party
    'constipated', 'normal_poop', 'diarrhea', 'great_poop',  # poop
    'withdrawal_sex', 'unprotected_sex', 'protected_sex', 'high_sex_drive',  # sex
    '3-6', '6-9', '0-3', '>9',  # sleep
    'conflict_social', 'supportive_social', 'sociable', 'withdrawn_social',  # social
    'ovulation_test_neg', 'ovulation_test_pos', 'pregnancy_test_neg', 'pregnancy_test_pos',  # test
]

list_of_symptoms = symptoms_of_interest + other_symptoms
ordered_symptoms = {s: i for i, s in enumerate(list_of_symptoms)}
symptoms_of_interest_dict = {code:symptom for symptom, code in ordered_symptoms.items() if code < 16}
training_columns = list_of_symptoms + ['day_in_cycle', 'absolute_day', 'period']


# ============  Feature engineer for the cycle ============
def expand_cycle(cycle):
    dates = pd.date_range(start=cycle.cycle_start, periods=cycle.cycle_length).tolist()
    period = np.zeros(cycle.cycle_length, dtype=np.int8)
    period[:cycle.period_length] = 1
    day_in_cycle = np.arange(1, cycle.cycle_length + 1, dtype=np.int8)

    index = pd.MultiIndex.from_tuples(
        tuples=list(zip([cycle.user_id] * int(cycle.cycle_length), dates)),
        names=["user_id", "date"]
    )

    return pd.DataFrame(
        data=list(zip([cycle.cycle_id] * int(cycle.cycle_length), day_in_cycle, period)),
        index=index,
        columns=['cycle_id', 'day_in_cycle', 'period']
    )


def expand_cycles(cycles):
    cycles_processed_backup = pj(data_dir, "cycles_processed.pkl.gz")
    try:
        cycles_processed = joblib.load(cycles_processed_backup)
    except:
        cycles_processed = pd.concat([expand_cycle(cycle) for _, cycle in cycles.iterrows()])
        joblib.dump(cycles_processed, cycles_processed_backup)

    return cycles_processed


# ========  Feature engineer for tracked symptoms =========
def process_tracking(tracking):
    # One hot encode the symptoms
    tracking_processed = pd.get_dummies(
        tracking[["user_id", "date", "symptom"]],
        columns=['symptom'], prefix='', prefix_sep=''
    )

    # Aggregate symptoms per day
    return tracking_processed.groupby(['user_id', 'date']).sum()[list_of_symptoms]


# ===============  Merging all the features ===============
def get_features(split=True, force=False):

    features_backup = pj(data_dir, 'features.pkl.gz')

    # Try to load from memory if already computed
    if os.path.exists(features_backup) and not force:
        features = joblib.load(features_backup)
    else:
        # Expand cycles so that there is a line per date (active or not) with a boolean indicator of period
        cycles_processed = expand_cycles(cycles)

        # Expand tracking so that there is a line per date (active or not) with a one hot encoded symtoms
        tracking_processed = process_tracking(tracking)

        # Merge cycles and tracking information
        features = pd.merge(cycles_processed, tracking_processed, left_index=True, right_index=True, how='outer').fillna(0)

        # Find the first day the user started using the app
        features = pd.merge(
            features,
            cycles.groupby('user_id')\
                  .agg({'cycle_start': {'first_use': 'min'}})\
                  .reset_index()\
                  .set_index('user_id')['cycle_start'],
            left_index=True,
            right_index=True
        )

        # Find the absolute day for each row from the day the user started using the app
        absolute_day = (features.reset_index().date.dt.date - features.reset_index().first_use.dt.date).dt.days + 1
        absolute_day.index = features.index
        features['absolute_day'] = absolute_day
        # Keep only the columns needed by the RNN
        features = features[training_columns]

        # Make a copy to speed up development iterations
        joblib.dump(features, features_backup)

        # This saves memory, I think...
        del tracking_processed
        del cycles_processed

    if split:
        # Do a train/test split of the data
        train_users = users.user_id.sample(frac=0.8)
        features = features.reset_index()
        df_train = features[features.user_id.isin(train_users)][training_columns]
        df_test = features[~features.user_id.isin(train_users)][training_columns]
        return df_train, df_test
    else:
        return features


# =========== Prepare data for model predictions ==========
def prepare_data_for_prediction(features=None, maxlen=90):
    """Clip the training dataset to maxlen days per user

    The LSTM predictor uses only the last maxlen days per user for predicting.
    """

    def get_user_daterange_index(user_id, max_date, length):
        """Return a list of tuples with [user_id] x list of 'length' last days.

        The list of tuples will be used to reconstruct an index with only the
        the previous 90 dates per user
        """
        dates = pd.date_range(end=max_date, periods=length)
        return list(zip([user_id] * len(dates), dates))

    if features is None:
        features = get_features(split=False)

    # Look up for the last day of activity per user
    cycles_processed = expand_cycles(cycles)
    day_maxs = cycles_processed.reset_index() \
        .groupby("user_id") \
        .agg({
        'date': {'max_date': 'max'}
    })['date']

    # Get dates for all users
    index_tuples = []
    for user_id, max_date in day_maxs.iterrows():
        index_tuples.extend(get_user_daterange_index(user_id, max_date.iloc[0], maxlen))

    # Construct the index with the last dates per user
    index = pd.MultiIndex.from_tuples(
        tuples=index_tuples,
        names=["user_id", "date"]
    )

    # Reindex and fill with 0 for women with less than 'maxlen' days of activity
    return features.reindex(index, fill_value=0)
