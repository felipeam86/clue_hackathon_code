from os.path import join as pj
import os
import pandas as pd
import numpy as np

MAXLEN = 90

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

other_symptoms = set(tracking.symptom.unique()) - set(symptoms_of_interest)
ordered_symptoms = {s:i for i, s in enumerate(symptoms_of_interest + list(other_symptoms))}
symptoms_of_interest_dict = {code:symptom for symptom, code in ordered_symptoms.items() if code < 16}
N_symptoms = len(ordered_symptoms)

tracking['symptom_code'] = tracking.symptom.map(lambda s: ordered_symptoms[s])

def transform_user(user_id):
    out = tracking[tracking.user_id == user_id]  
    out.loc[:, 'absolute_day'] = (out.date.dt.date - out.date.dt.date.min()).dt.days + 1
    out = pd.merge(out, cycles[cycles.user_id == user_id], on=["cycle_id", "user_id"])

    out_sequence = np.zeros(
        (
            out.absolute_day.max(),
            N_symptoms + 1
        )
    )
    out_sequence[:, -1] = range(1, out_sequence.shape[0] + 1)
    for _, row in out.iterrows():
        out_sequence[row['absolute_day'] - 1, row['symptom_code']] = 1
    return out_sequence


def transform_users(user_ids):
    return np.vstack(
        [transform_user(user_id)
         for user_id in user_ids]
    )

def load_sequence():
    print("Loading users data")
    out = tracking[["user_id","cycle_id","date","day_in_cycle","symptom"]]
    out.set_index("user_id",inplace=True)
    out.index.name = None

    women = tracking[["user_id","date"]]
    mins_maxs = pd.groupby(women,by="user_id").min()
    mins_maxs.index.name = None
    mins_maxs = mins_maxs.merge(pd.groupby(women,by="user_id").max(),left_index=True,right_index=True)
    mins_maxs.columns = ["min_date","max_date"]

    out2 = out.merge(mins_maxs,how='left',left_index=True,right_index=True,)
    out2["absolute_day"] = (out2.date - out2.min_date).dt.days +1
    out2.reset_index(inplace=True)
    out2.columns = ["user_id","cycle_id","date","day_in_cycle","symptom_code","min_date","max_date","absolute_day"]
    
    print("Reformatting users data")
    pivot = pd.pivot_table(out2,
                            index=["user_id","date"],#,"cycle_id","day_in_cycle"],
                            columns="symptom_code",
                            values="absolute_day",
                            fill_value=0,
                            aggfunc=lambda x: 1)

    date_list,user_id_list = [],[]
    for row in mins_maxs.iterrows():
        (min,max) = row[1]
        min = max - pd.Timedelta(MAXLEN-1,unit="D")
        date_list.extend(pd.date_range(min,max).tolist())
        user_id_list.extend([row[0] for _ in range(MAXLEN)])

    multi_idx = pd.MultiIndex.from_arrays([user_id_list,date_list],names = ["user_id","date"])
    rnn_input = pivot.reindex(multi_idx, fill_value=0).reset_index(1,drop=True)

    symptoms_of_interest = [
        'happy', 'pms', 'sad', 'sensitive_emotion',
        'energized', 'exhausted', 'high_energy', 'low_energy',
        'cramps', 'headache', 'ovulation_pain', 'tender_breasts',
        'acne_skin', 'good_skin', 'oily_skin', 'dry_skin'
    ]

    other_symptoms = set(tracking.symptom.unique()) - set(symptoms_of_interest)
    ordered_symptoms = {s:i for i, s in enumerate(symptoms_of_interest + list(other_symptoms))}
    symptoms_of_interest_dict = {code:symptom for symptom, code in ordered_symptoms.items() if code < 16}
    N_symptoms = len(ordered_symptoms)

    ordered_symptoms = list(symptoms_of_interest)
    ordered_symptoms.extend(list(other_symptoms))
    rnn_input = rnn_input.loc[:,ordered_symptoms]
    rnn_input["pregnancy_test_pos"] = 0

    return rnn_input

def transform_user(user_id):
    out = tracking[tracking.user_id == user_id]  
    out.loc[:, 'absolute_day'] = (out.date.dt.date - out.date.dt.date.min()).dt.days + 1
    out = pd.merge(out, cycles[cycles.user_id == user_id], on=["cycle_id", "user_id"])

    out_sequence = np.zeros(
        (
            out.absolute_day.max(),
            N_symptoms + 1
        )
    )
    out_sequence[:, -1] = range(1, out_sequence.shape[0] + 1)
    for _, row in out.iterrows():
        out_sequence[row['absolute_day'] - 1, row['symptom_code']] = 1
    return out_sequence


def transform_users(user_ids):
    return np.vstack(
        [transform_user(user_id)
         for user_id in user_ids]
    )

def get_sample_of_users(n, min_tracking_count=20):
    symptom_tracking_count = tracking.user_id.value_counts()
    interesting_users = symptom_tracking_count[symptom_tracking_count > min_tracking_count]

    return list(
        interesting_users.sample(n).index
    )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-N_users', default=10, type=int,
        help='How many users to sequence')
    parser.add_argument(
        '-maxlen', default=90, type=int,
        help='How many days to predict')

    args = parser.parse_args()

    sample_of_users = get_sample_of_users(args.N_users)
    sequence = load_sequence()
