import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
import neurokit2 as nk2
from tqdm import tqdm

import nkcopy
#import frequency_analysis


def ecg_process_custom(ecg_signal, sampling_rate=1000, method='neurokit'):
    try:
        output = nk2.ecg_process(ecg_signal, sampling_rate, method)
    except:
        output = np.nan
    return output


def extract_features(ecg_df, save=True, mode='train'): ## feature amount = 42
    '''
    input: raw X_train_df
    '''
    ecg_df = ecg_df.drop('id', axis=1)
    ecg_df = ecg_df.drop('x17978', axis=1) # /n at the end of the df screws things up

    features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                              'madNN', 'mcvNN', 'pNN50', 'pNN20']
    additional_peaks = {'ppeaks':'ECG_P_Peaks', 'qpeaks':'ECG_Q_Peaks', 'speaks':'ECG_S_Peaks', 'tpeaks':'ECG_T_Peaks'}

    # MAIN FEATURES, INITIALIZING #
    print('-- EXTRACTING MAIN FEATURES --')
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
                                'filtered': values.apply(lambda x: x['filtered']),
                                'templates': values.apply(lambda x: x['templates'])})
    print('-- VALUES EXTRACTION DONE --')
    #values_nk2 = ecg_df.apply(lambda x: ecg_process_custom(x.dropna(), sampling_rate=300)[0], axis=1)
    values_nk2 = ecg_df.apply(lambda x: ecg_process_custom(x.dropna(), sampling_rate=300), axis=1)
    values_nk2 = values_nk2.apply(lambda x: x[0] if x==x else np.nan)
    for k,v in additional_peaks.items():
        peak_name = k[0]
        features_df[k] = values_nk2.apply(lambda x: np.array(x[v]) if type(x)==pd.core.frame.DataFrame else np.nan) # problem in this line change x==x
        features_df[k] = features_df[k].apply(lambda x: np.where(x == 1)[0] if type(x)==np.ndarray else np.nan)
        print('----->'+peak_name+' done.')
    print('-- VALUES_NK2 EXTRACTION DONE --')


    # R PEAK FEATURES
    features_df['R_peaks'] = features_df.apply(lambda x: x['filtered'][x['rpeaks']], axis=1)

    features_df['mean_rvalues'] = features_df.apply(lambda x: np.mean(x['R_peaks']), axis=1)
    features_df['min_rvalues'] = features_df.apply(lambda x: np.min(x['R_peaks']), axis=1)
    features_df['max_rvalues'] = features_df.apply(lambda x: np.max(x['R_peaks']), axis=1)
    features_df['std_rvalues'] = features_df.apply(lambda x: np.std(x['R_peaks']), axis=1)
    features_df['median_rvalues'] = features_df.apply(lambda x: np.median(x['R_peaks']), axis=1)
    print('-- R PEEK EXTRACTION DONE --')

    # ADDITIONAL PEAK FEATURES
    for k in additional_peaks.keys():
        features_df[k] = features_df.apply(lambda x: x['filtered'][x[k]] if type(x[k])==np.ndarray else np.nan, axis=1)

        peak_name = k[0]
        features_df['mean_'+peak_name+'values'] = features_df.apply(lambda x: np.mean(x[k]) if (type(x[k])==np.ndarray and len(x[k])!=0) else np.nan, axis=1)
        features_df['min_'+peak_name+'values'] = features_df.apply(lambda x: np.min(x[k]) if (type(x[k])==np.ndarray and len(x[k])!=0) else np.nan, axis=1)
        features_df['max_'+peak_name+'values'] = features_df.apply(lambda x: np.max(x[k]) if (type(x[k])==np.ndarray and len(x[k])!=0) else np.nan, axis=1)
        features_df['std_'+peak_name+'values'] = features_df.apply(lambda x: np.std(x[k]) if (type(x[k])==np.ndarray and len(x[k])!=0) else np.nan, axis=1)
        features_df['median_'+peak_name+'values'] = features_df.apply(lambda x: np.median(x[k]) if (type(x[k])==np.ndarray and len(x[k])!=0) else np.nan, axis=1)
        print('----->' + peak_name + ' done.')
    print('-- OTHER PEEKS EXTRACTION DONE --')


    # POWER
    features_df['power'] = features_df['filtered'].apply(lambda x: np.sum(np.square(x)) / x.shape[0])

    # CARDIADIC CYCLES
    # features_df['Cardiadic_Cycles'] = features_df['templates'].apply(lambda x: x)
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
    print('-- CARDIADIC CYCLES EXTRACTION DONE --')

    # HRV FEATURES
    features_df['hrv_time_features'] = features_df['rpeaks'].apply(lambda x: nkcopy.ecg_hrv(rpeaks=x,
                                                                                     sampling_rate=300,
                                                                                     hrv_features='time'))
    for name in features_names_timehvr:
        features_df[name] = features_df['hrv_time_features'].apply(lambda x: x[name])
    print('-- HRV EXTRACTION DONE --')

    # FINALIZE / SAVE
    features_df = features_df.drop(['rpeaks', 'filtered', 'templates', 'R_peaks', 'hrv_time_features'], axis=1)
    features_df = features_df.drop(list(additional_peaks.keys()), axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_'+mode+'_'+str(numberOfFeatures)+'.csv', index=False)
        print('-- FEATURES SAVED --')

    return features_df

# TRAIN
X_train_df = pd.read_csv('../raw/X_train.csv')
extract_features(X_train_df)

# # TEST
# X_test_df = pd.read_csv('../raw/X_test.csv')
# extract_features(X_test_df, mode='test')



