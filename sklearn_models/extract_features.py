## INCLUDES AMPLITUDE FEATURES, FREQUENCY FEATURES, EMG ONSET FEATURES ##
# Insight: if an epoch contains EMG onset, it's in class 1 %100 of the time

import numpy as np
import pandas as pd
from biosppy.signals import eeg, emg
#import neurokit as nk
from tqdm import tqdm
import frequency_analysis

freq_bands_general = {'delta':[0,0.04],
					  'theta':[0.04, 0.08],
					  'alpha_low':[0.08, 0.1],
					  'alpha_high':[0.1, 0.13],
					  'beta':[0.13, 0.25],
					  'gamma':[0.25, 0.5]}

def extract_features(eeg1_df, eeg2_df, emg_df, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    eeg1_df = eeg1_df.drop('Id', axis=1)
    eeg2_df = eeg2_df.drop('Id', axis=1)
    emg_df = emg_df.drop('Id', axis=1)

    features_names_freq = ['delta', 'theta', 'alpha_low', 'alpha_high', 'beta', 'gamma',
                           'Total', 'corr_max', 'corr_min', 'corr1', 'corr2', 'corr3']
    ## can add ['extra1', 'extra2', 'extra3', 'extra4', 'extra5']


    # MAIN FEATURES, INITIALIZING #
    values_eeg1 = eeg1_df.apply(lambda x: eeg.eeg(np.array(x).reshape(512,1), sampling_rate=128, show=False), axis=1)
    values_eeg2 = eeg2_df.apply(lambda x: eeg.eeg(np.array(x).reshape(512,1), sampling_rate=128, show=False), axis=1)
    values_emg = emg_df.apply(lambda x: emg.emg(np.array(x), show=False), axis=1) # no sampling_rate because it causes issues :)

    features_df = pd.DataFrame({'filtered_eeg1': values_eeg1.apply(lambda x: x['filtered'].reshape(512,)),
                                'filtered_eeg2': values_eeg2.apply(lambda x: x['filtered'].reshape(512,)),
                                'filtered_emg': values_emg.apply(lambda x: x['filtered'].reshape(512,)),
                                'emg_onsets': values_emg.apply(lambda x: x['onsets'])})


    # AMPLITUDE FEATURES
    df_names = ['eeg1', 'eeg2', 'emg']
    for name in df_names:
        features_df['mean_'+name] = features_df.apply(lambda x: np.mean(x['filtered_'+name]), axis=1)
        features_df['min_' + name] = features_df.apply(lambda x: np.min(x['filtered_' + name]), axis=1)
        features_df['max_' + name] = features_df.apply(lambda x: np.max(x['filtered_' + name]), axis=1)
        features_df['std_' + name] = features_df.apply(lambda x: np.std(x['filtered_' + name]), axis=1)
        features_df['median_' + name] = features_df.apply(lambda x: np.median(x['filtered_' + name]), axis=1)


    # FREQUENCY FEATURES
    freq_bands = ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']  # extracted using biosppy
    for band in freq_bands:
        features_df[band+'_eeg1'] = values_eeg1.apply(lambda x: np.mean(x[band]))
        features_df[band+'_eeg2'] = values_eeg2.apply(lambda x: np.mean(x[band]))

    features_df['freq_features_eeg1'] = features_df['filtered_eeg1'].apply(lambda x: frequency_analysis.get_frequency_features(sig=x))
    features_df['freq_features_eeg2'] = features_df['filtered_eeg2'].apply(lambda x: frequency_analysis.get_frequency_features(sig=x))

    for name in features_names_freq:
        features_df[name+'_2_eeg1'] = features_df['freq_features_eeg1'].apply(lambda x: x[name])
        features_df[name + '_2_eeg2'] = features_df['freq_features_eeg2'].apply(lambda x: x[name])


    # EMG ONSET FEATURES
    # 1) try finding onsets for individual epochs
    # 2) find onsets for individual patients, extract epoch indexes
    features_df['consists_emg_onset'] = features_df['emg_onsets'].apply(lambda x: 0 if len(x)==0 else 1)


    # FINALIZE / SAVE
    features_df = features_df.drop(['filtered_eeg1', 'filtered_eeg2', 'filtered_emg',
                                    'emg_onsets', 'freq_features_eeg1', 'freq_features_eeg2'], axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_'+mode+'_'+str(numberOfFeatures)+'.csv', index=False)

    return features_df

# TRAIN
train_eeg1_df = pd.read_csv('../raw/train_eeg1.csv')
train_eeg2_df = pd.read_csv('../raw/train_eeg2.csv')
train_emg_df = pd.read_csv('../raw/train_emg.csv')
extract_features(train_eeg1_df, train_eeg2_df, train_emg_df)

# # TEST
# test_eeg1_df = pd.read_csv('../raw/test_eeg1.csv')
# test_eeg2_df = pd.read_csv('../raw/test_eeg2.csv')
# test_emg_df = pd.read_csv('../raw/test_emg.csv')
# extract_features(test_eeg1_df, test_eeg2_df, test_emg_df, mode='test')



