import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
#import neurokit as nk
from tqdm import tqdm

import nkcopy
#import frequency_analysis



def extract_features(ecg_df, numberOfFeatures = 27, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    ecg_df = ecg_df.drop('id', axis=1)

    features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                              'madNN', 'mcvNN', 'pNN50', 'pNN20']

    features_names_freqhvr = ['ULF', 'VLF', 'LF', 'HF', 'VHF', 'Total', 'extra1', 'extra2',
                              'extra3', 'extra4', 'extra5', 'corr_max', 'corr_min', 'corr1', 'corr2', 'corr3']

    number_of_samples = ecg_df.values.shape[0]

    extracted_features = np.empty([number_of_samples, numberOfFeatures])
    pbar = tqdm(total=number_of_samples)
    pbar.update(0)

    for i in range(number_of_samples):
        pbar.update(1)
        #if i % 100 == 0:
        #    print('Iteration {i} done'.format(i=i))
        features = np.empty([1, 0])

        currentPatient = ecg_df.iloc[i].dropna().values

        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(currentPatient,
                                                                                           sampling_rate=300,
                                                                                           show=False)
        #templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks=rpeaks, sampling_rate=300)


        # add mean, max, min, median, std of r amplitudes
        R_Peaks = filtered[rpeaks]

        mean_rvalues = np.mean(R_Peaks)
        min_rvalues = np.min(R_Peaks)
        max_rvalues = np.max(R_Peaks)
        std_rvalues = np.std(R_Peaks)
        median_rvalues = np.median(R_Peaks)

        means_of_r = np.empty([1, 5])
        means_of_r[0, :] = [mean_rvalues, min_rvalues, max_rvalues, std_rvalues, median_rvalues]
        features = np.append(features, means_of_r, axis=1)


        # add power (whatever this is)
        power = np.sum(np.square(filtered)) / filtered.shape[0]
        features = np.append(features, power.reshape(1, -1), axis=1)


        # add the mean, mean-max, mean-min, mean-std, mean-median of cardiac cycles
        Cardiadic_Cycles = pd.DataFrame(templates)

        mean = Cardiadic_Cycles.mean(axis=0).mean()
        mean_max = Cardiadic_Cycles.max(axis=0).mean()
        mean_min = Cardiadic_Cycles.min(axis=0).mean()
        mean_median = Cardiadic_Cycles.median(axis=0).mean()
        mean_std = Cardiadic_Cycles.std(axis=0).mean()

        max_min = Cardiadic_Cycles.min(axis=0).max()
        min_min = Cardiadic_Cycles.min(axis=0).min()

        max_max = Cardiadic_Cycles.max(axis=0).max()
        min_max = Cardiadic_Cycles.max(axis=0).min()

        max_std = Cardiadic_Cycles.std(axis=0).max()
        min_std = Cardiadic_Cycles.std(axis=0).min()

        to_add = np.empty([1, 11])
        to_add[0, :] = [mean, mean_max, mean_min, mean_median, mean_std, max_min, min_min, max_max, min_max, max_std,
                        min_std]

        features = np.append(features, to_add, axis=1)


        ## Addition of frequency features (numberOfFeatures 17 -> 33)
        #hvr_freq_features = frequency_analysis.get_frequency_features(templates)
        ## add all frequency hvr to array features
        #for feature in features_names_freqhvr:
        #    features = np.append(features, hvr_freq_features[feature])


        # Addition of time hvr features (numberOfFeatures 17 -> 27)
        #hvr_time_features = nk.ecg_hrv(rpeaks=rpeaks, sampling_rate=300, hrv_features='time')
        hvr_time_features = nkcopy.ecg_hrv(rpeaks=rpeaks, sampling_rate=300, hrv_features='time')
        ## add all time hvr to array features
        for feature in features_names_timehvr:
            features = np.append(features, hvr_time_features[feature])


        extracted_features[i, :] = features

    if save:
        extracted_features_df = pd.DataFrame(extracted_features)
        pd.to_csv('features_'+mode+'_'+str(numberOfFeatures)+'.csv', index=False)

    return extracted_features

X_train_df = pd.read_csv('../raw/X_train.csv')
extract_features(X_train_df)



