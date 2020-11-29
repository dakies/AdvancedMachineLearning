## INCLUDES R PEAKS, RR INTERVALS, FREQUENCY FEATURES ##

import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
#import neurokit as nk
from tqdm import tqdm

import nkcopy
import frequency_analysis


def extract_features(ecg_df, save=True, mode='train'): ## feature amount = 43
    '''
    input: raw X_train_df
    '''
    print('Cleaning dataset')
    ecg_df = ecg_df.drop('id', axis=1)
    ecg_df = ecg_df.drop('x17978', axis=1) # /n at the end of the df screws thing up
    # Remove initial noisy timesteps from signal
    for i in range(500):
        ecg_df = ecg_df.drop('x' + str(i), axis=1)

    features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                              'madNN', 'mcvNN', 'pNN50', 'pNN20']

    features_names_freqhvr = ['ULF', 'VLF', 'LF', 'HF', 'VHF', 'Total', 'extra1', 'extra2',
                              'extra3', 'extra4', 'extra5', 'corr_max', 'corr_min', 'corr1', 'corr2', 'corr3']

    # MAIN FEATURES, INITIALIZING #
    print('Extracting main features')
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
                                'filtered': values.apply(lambda x: x['filtered']),
                                'templates': values.apply(lambda x: x['templates']),
                                'heart_rate': values.apply(lambda x: x['heart_rate'])})

    # R PEAK FEATURES
    print('Extracting r and rate features')
    features_df['R_peaks'] = features_df.apply(lambda x: x['filtered'][x['rpeaks']], axis=1)

    def check_nan_func(func, vals):
        return func(vals) if (type(vals) == np.ndarray and len(vals) != 0) else np.nan

    fields = {'rvalues': 'R_peaks', 'rate': 'heart_rate'}
    for key, val in fields.items():
        features_df['mean_'+key] = features_df.apply(lambda x: check_nan_func(np.mean, x[val]), axis=1)
        features_df['min_'+key] = features_df.apply(lambda x: check_nan_func(np.min, x[val]), axis=1)
        features_df['max_'+key] = features_df.apply(lambda x: check_nan_func(np.max, x[val]), axis=1)
        features_df['std_'+key] = features_df.apply(lambda x: check_nan_func(np.std, x[val]), axis=1)
        features_df['median_'+key] = features_df.apply(lambda x: check_nan_func(np.median, x[val]), axis=1)

    # POWER
    features_df['power'] = features_df['filtered'].apply(lambda x: np.sum(np.square(x)) / x.shape[0])

    # CARDIADIC CYCLES
    # features_df['Cardiadic_Cycles'] = features_df['templates'].apply(lambda x: x)
    print('Extracting other features')
    features_df['mean'] = features_df['templates'].apply(lambda x: np.mean(np.mean(x, axis=0)))
    features_df['mean_max'] = features_df['templates'].apply(lambda x: np.mean(np.max(x, axis=0)))
    features_df['mean_min'] = features_df['templates'].apply(lambda x: np.mean(np.min(x, axis=0)))
    features_df['mean_median'] = features_df['templates'].apply(lambda x: np.mean(np.median(x, axis=0)))
    features_df['mean_std'] = features_df['templates'].apply(lambda x: np.mean(np.std(x.astype(np.float), axis=0)))

    features_df['max_min'] = features_df['templates'].apply(lambda x: np.max(np.min(x, axis=0)))
    features_df['min_min'] = features_df['templates'].apply(lambda x: np.min(np.min(x, axis=0)))

    features_df['max_max'] = features_df['templates'].apply(lambda x: np.max(np.max(x, axis=0)))
    features_df['min_max'] = features_df['templates'].apply(lambda x: np.min(np.max(x, axis=0)))

    features_df['min_std'] = features_df['templates'].apply(lambda x: np.min(np.std(x.astype(np.float), axis=0)))
    features_df['max_std'] = features_df['templates'].apply(lambda x: np.max(np.std(x.astype(np.float), axis=0)))

    # HRV FEATURES
    print('hrv features')
    features_df['hrv_time_features'] = features_df['rpeaks'].apply(lambda x: nkcopy.ecg_hrv(rpeaks=x,
                                                                                     sampling_rate=300,
                                                                                     hrv_features='time'))
    for name in features_names_timehvr:
        features_df[name] = features_df['hrv_time_features'].apply(lambda x: x[name])

    # FREQUENCY FEATURES
    print('freq features')
    features_df['hrv_freq_features'] = features_df['templates'].apply(lambda x: frequency_analysis.get_frequency_features(heartbeats=x))
    for name in features_names_freqhvr:
        features_df[name] = features_df['hrv_freq_features'].apply(lambda x: x[name])

    # FINALIZE / SAVE
    features_df = features_df.drop(['rpeaks', 'filtered', 'templates', 'heart_rate', 'R_peaks', 'hrv_time_features', 'hrv_freq_features'], axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_'+mode+'_'+str(numberOfFeatures)+'.csv', index=False)
        print('Saved dataset!')   
    return features_df

# # TRAIN
# if debug:
#     rng = np.random.default_rng(seed=1)
#     skip = rng.choice(np.arange(1, 5118, step=1), size=5110, replace=False)
#     X_train_df = pd.read_csv('raw/X_train.csv', skiprows=skip)
#     Y_train_df = pd.read_csv('raw/y_train.csv', skiprows=skip)
#     Y_train_df.to_csv('./features/y_train.csv', index=False, header=['id', 'y'], compression=None)
# else:
#     X_train_df = pd.read_csv('raw/X_train.csv')
# extract_features(X_train_df, save=True)

# TRAIN
X_train_df = pd.read_csv('../raw/X_train.csv')
print('Loaded train data')
extract_features(X_train_df)

# TEST
X_test_df = pd.read_csv('../raw/X_test.csv')
print('Loaded test data')
extract_features(X_test_df, mode='test')
