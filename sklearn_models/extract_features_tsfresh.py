# INCLUDES TSFRESH FEATURES #

import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
import neurokit2 as nk2
from tqdm import tqdm

import tsfresh as ts
from tsfresh.feature_extraction import EfficientFCParameters

def extract_features(ecg_df, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    ecg_df = ecg_df.drop('id', axis=1)
    ecg_df = ecg_df.drop('x17978', axis=1) # /n at the end of the df screws things up

    # MAIN FEATURES, INITIALIZING #
    print('-- EXTRACTING MAIN FEATURES --')
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'filtered': values.apply(lambda x: x['filtered'])})
    print('-- VALUES EXTRACTION DONE --')


    # TS DATAFRAME CREATION #
    length_ts = len(features_df)
    end_dfs_ts = []
    pbar = tqdm(total=length_ts)
    pbar.update(0)
    for i in range(length_ts):
        pbar.update(1)
        id = i * np.ones(len(features_df.iloc[i]['filtered']), dtype=int)
        times = np.arange(len(features_df.iloc[i]['filtered']), dtype=int)
        data = features_df.iloc[i]['filtered']#.reshape(len(features_df.iloc[i]['filtered']), 1)
        sample_ts = pd.DataFrame({'id':id,
                                  'time':times,
                                  'data':data})
        end_dfs_ts.append(sample_ts)
    print('-- TS DATAFRAME CREATION DONE --')

    # TS FEATURE EXTRACTION #
    input_ts = pd.concat(end_dfs_ts, axis=0)
    settings = EfficientFCParameters()
    extracted_features_ts = ts.extract_features(input_ts, default_fc_parameters=settings, column_id="id",
                                                          column_sort="time", column_value='data')
    print('-- TS DATAFRAME CREATION DONE --')

    # FINALIZE / SAVE
    if save:
        extracted_features_ts.to_csv('./features/tsfresh_features_'+mode+'.csv', index=False)
        print('-- FEATURES SAVED --')

    return extracted_features_ts

# TRAIN
X_train_df = pd.read_csv('../raw/X_train.csv')
extract_features(X_train_df)

# # TEST
# X_test_df = pd.read_csv('../raw/X_test.csv')
# extract_features(X_test_df, mode='test')





