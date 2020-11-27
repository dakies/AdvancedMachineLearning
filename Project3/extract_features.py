import numpy as np
import pandas as pd
from neurokit2 import ecg_process
from neurokit2 import hrv


# import nkcopy


# import frequency_analysis


def extract_features(ecg_df, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    features_names_hvr = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN',
                          'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN',
                          'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF', 'HRV_VLF',
                          'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
                          'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
                          'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
                          'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
                          'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
                          'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn', 'HRV_SampEn']

    # MAIN FEATURES, INITIALIZING #
    # values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    # features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
    #                            'filtered': values.apply(lambda x: x['filtered']),
    #                            'templates': values.apply(lambda x: x['templates'])})
    # Add peaks: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"
    values = ecg_df.apply(lambda x: ecg_process(x.dropna(), method="hamilton", sampling_rate=300), axis=1)
    features_df = pd.DataFrame({'signal': values.apply(lambda x: x[0]),
                                'info': values.apply(lambda x: x[1])})
    # features_df['ECG_P_Peaks'] = values.apply(lambda x: x['ECG_P_Peaks'])
    # features_df['ECG_Q_Peaks'] = values.apply(lambda x: x['ECG_Q_Peaks'])
    # features_df['ECG_S_Peaks'] = values.apply(lambda x: x['ECG_S_Peaks'])
    # features_df['ECG_T_Peaks'] = values.apply(lambda x: x['ECG_T_Peaks'])
    # features_df['ECG_P_Onsets'] = values.apply(lambda x: x['ECG_P_Onsets'])
    # features_df['ECG_T_Offsets'] = values.apply(lambda x: x['ECG_T_Offsets'])

    peaks = ['ECG_R_Peaks', 'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks', 'ECG_P_Onsets', 'ECG_T_Offsets']
    for i in peaks:
        print(i)
        features_df['val_' + i] = features_df.apply(
            lambda x: np.array(x['signal']['ECG_Clean'].loc[x['signal'][i] == 1]), axis=1)
        features_df['mean_' + i] = features_df.apply(lambda x: np.mean(x['val_' + i]), axis=1)
        features_df['min_' + i] = features_df.apply(lambda x: np.min(x['val_' + i]), axis=1)
        features_df['max_' + i] = features_df.apply(lambda x: np.max(x['val_' + i]), axis=1)
        features_df['std_' + i] = features_df.apply(lambda x: np.std(x['val_' + i]), axis=1)
        features_df['median_' + i] = features_df.apply(lambda x: np.median(x['val_' + i]), axis=1)

    # Todo: Analyze time series with tsfresh
    # POWER
    features_df['power'] = features_df.apply(
        lambda x: np.sum(np.square(x['signal']['ECG_Clean'])) / x['signal']['ECG_Clean'].shape[0], axis=1)

    ## CARDIADIC CYCLES
    ## features_df['Cardiadic_Cycles'] = features_df['templates'].apply(lambda x: x)
    # features_df['mean'] = features_df['templates'].apply(lambda x: np.mean(np.mean(x, axis=0)))
    # features_df['mean_median'] = features_df['templates'].apply(lambda x: np.mean(np.median(x, axis=0)))
    # features_df['mean_std'] = features_df['templates'].apply(lambda x: np.mean(np.std(x.astype(np.float), axis=0)))
    # features_df['min_std'] = features_df['templates'].apply(lambda x: np.min(np.std(x.astype(np.float), axis=0)))
    # features_df['max_std'] = features_df['templates'].apply(lambda x: np.max(np.std(x.astype(np.float), axis=0)))
    # Todo replace std with quality signal?

    # HRV FEATURES
    features_df['hrv_features'] = features_df.apply(lambda x: hrv(peaks=x['info'], sampling_rate=300), axis=1)
    for name in features_names_hvr:
        features_df[name] = features_df['hrv_features'].apply(lambda x: x[name])

    # FINALIZE / SAVE
    features_df = features_df.drop(['val_' + s for s in peaks], axis=1)
    features_df = features_df.drop(['hrv_features', 'signal', 'info'], axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('feat3/' + mode + '_' + str(numberOfFeatures) + '.csv', index=False)

    return features_df


debug = 0
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

extract_features(x_train, mode='train')

# # TEST
# X_test_df = pd.read_csv('../raw/X_test.csv')
# extract_features(X_test_df, mode='test')
