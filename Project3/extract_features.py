import biosppy.signals.ecg as ecg
import nkcopy
import numpy as np
import pandas as pd
# import neurokit as nk
from neurokit2.ecg import ecg_delineate


# import nkcopy


# import frequency_analysis


def extract_features(ecg_df, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                              'madNN', 'mcvNN', 'pNN50', 'pNN20']

    # MAIN FEATURES, INITIALIZING #
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
                                'filtered': values.apply(lambda x: x['filtered']),
                                'templates': values.apply(lambda x: x['templates'])})
    # Add peaks: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"
    values = features_df.apply(lambda x: ecg_delineate(
        x['filtered'], rpeaks=x['rpeaks'], sampling_rate=300)[1], axis=1)
    features_df['ECG_P_Peaks'] = values.apply(lambda x: x['ECG_P_Peaks'])
    features_df['ECG_Q_Peaks'] = values.apply(lambda x: x['ECG_Q_Peaks'])
    features_df['ECG_S_Peaks'] = values.apply(lambda x: x['ECG_S_Peaks'])
    features_df['ECG_T_Peaks'] = values.apply(lambda x: x['ECG_T_Peaks'])
    features_df['ECG_P_Onsets'] = values.apply(lambda x: x['ECG_P_Onsets'])
    features_df['ECG_T_Offsets'] = values.apply(lambda x: x['ECG_T_Offsets'])

    peaks = ['ECG_P_Peaks', 'rpeaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks', 'ECG_P_Onsets', 'ECG_T_Offsets']
    for i in peaks:
        print(i)
        features_df['val_' + i] = features_df.apply(lambda x: x['filtered'][x[i]], axis=1)
        features_df['mean_' + i] = features_df.apply(lambda x: np.mean(x['val_' + i]), axis=1)
        features_df['min_' + i] = features_df.apply(lambda x: np.min(x['val_' + i]), axis=1)
        features_df['max_' + i] = features_df.apply(lambda x: np.max(x['val_' + i]), axis=1)
        features_df['std_' + i] = features_df.apply(lambda x: np.std(x['val_' + i]), axis=1)
        features_df['median_' + i] = features_df.apply(lambda x: np.median(x['val_' + i]), axis=1)

    # Todo: Analyze time series with tsfresh
    # POWER
    features_df['power'] = features_df['filtered'].apply(lambda x: np.sum(np.square(x)) / x.shape[0])

    # CARDIADIC CYCLES
    # features_df['Cardiadic_Cycles'] = features_df['templates'].apply(lambda x: x)
    features_df['mean'] = features_df['templates'].apply(lambda x: np.mean(np.mean(x, axis=0)))
    features_df['mean_median'] = features_df['templates'].apply(lambda x: np.mean(np.median(x, axis=0)))
    features_df['mean_std'] = features_df['templates'].apply(lambda x: np.mean(np.std(x.astype(np.float), axis=0)))
    features_df['min_std'] = features_df['templates'].apply(lambda x: np.min(np.std(x.astype(np.float), axis=0)))
    features_df['max_std'] = features_df['templates'].apply(lambda x: np.max(np.std(x.astype(np.float), axis=0)))
    # Todo replace std with quality signal?

    # HRV FEATURES
    # Todo: neurkit2 has more HRV features
    features_df['hrv_time_features'] = features_df['rpeaks'].apply(lambda x: nkcopy.ecg_hrv(rpeaks=x,
                                                                                            sampling_rate=300,
                                                                                            hrv_features='time'))
    for name in features_names_timehvr:
        features_df[name] = features_df['hrv_time_features'].apply(lambda x: x[name])

    # FINALIZE / SAVE
    features_df = features_df.drop(['rpeaks', 'filtered', 'templates', 'R_peaks', 'hrv_time_features'], axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_' + mode + '_' + str(numberOfFeatures) + '.csv', index=False)

    return features_df


debug = 1
if debug:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Debug mode activated')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Chose some random samples to load
    rng = np.random.default_rng(seed=1)
    skip = rng.choice(np.arange(1, 5118, step=1), size=5107, replace=False)
    # skip = np.arange(1, 5118, step=1).tolist()
    x_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id', skiprows=skip)
else:
    x_test = pd.read_csv('raw/X_test.csv', sep=',', index_col='id')
    x_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id')
    print('Data loaded')

# TRAIN
# X_train_df = pd.read_csv('Project3/raw/X_train.csv')

extract_features(x_train)

# # TEST
# X_test_df = pd.read_csv('../raw/X_test.csv')
# extract_features(X_test_df, mode='test')
