## INCLUDES AMPLITUDE FEATURES, FREQUENCY FEATURES, EMG ONSET FEATURES ##
# Insight: if an epoch contains EMG onset, it's in class 1 %100 of the time
# Idea: combine 5 consecutive epochs (since there is coherence between epochs), predict for the middle epoch

import numpy as np
import pandas as pd
from biosppy.signals import eeg, emg
#import neurokit as nk
from tqdm import tqdm
import frequency_analysis
import utils

freq_bands_general = {'delta':[0,0.04],
					  'theta':[0.04, 0.08],
					  'alpha_low':[0.08, 0.1],
					  'alpha_high':[0.1, 0.13],
					  'beta':[0.13, 0.25],
					  'gamma':[0.25, 0.5]}

def extract_features(eeg1_df, eeg2_df, emg_df, save=True, mode='train', combine_5_epochs=True):
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
    # 1) finding onsets for individual epochs
    features_df['consists_emg_onset'] = features_df['emg_onsets'].apply(lambda x: 0 if len(x) == 0 else 1)
    # 2) finding onsets for individual patients, extract epoch indexes
    number_of_patients = int(len(emg_df)/21600)
    onset_features_patient = []
    for i in range(number_of_patients):
        patient_emg = emg_df.values.reshape(-1)
        patient_emg = patient_emg[i*21600*512:(i+1)*21600*512]
        ts_emg_all, filtered_emg_all, onsets_emg_all = emg.emg(patient_emg, show=False)
        onset_indexes = np.array(onsets_emg_all / 512, dtype=int)
        onset_indexes = np.unique(onset_indexes)
        for j in range(21600):
            if j in onset_indexes:
                onset_features_patient.append(1)
            else:
                onset_features_patient.append(0)
    features_df['consists_emg_onset_patient'] = np.array(onset_features_patient)


    # COMBINING 5 EPOCH FEATURES / DROP UNNECESSARY COLUMNS
    features_df = features_df.drop(['filtered_eeg1', 'filtered_eeg2', 'filtered_emg',
                                    'emg_onsets', 'freq_features_eeg1', 'freq_features_eeg2'], axis=1)
    if combine_5_epochs:
        features_df_5_epochs = utils.combine_5_epochs(features_df)


    # FINALIZE / SAVE
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_'+mode+'_'+str(numberOfFeatures)+'.csv', index=False)
        if combine_5_epochs:
            features_df_5_epochs.to_csv('./features/features_5e_' + mode + '_' + str(numberOfFeatures*5) + '.csv', index=False)

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



