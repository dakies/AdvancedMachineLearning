import itertools
import time
from time import time

import neurokit2 as nk
import numpy as np
import pandas as pd
from biosppy import ecg
from tsfresh.feature_extraction import feature_calculators as fc


def switch(x):
    if abs(np.max(x)) < abs(np.min(x)):
        return 1
    else:
        return -1


def extract_template_info(template):
    peak = np.amax(template)
    idx_peak = np.argmax(template)

    if idx_peak == 0:
        low_before = peak
        idx_low_before = idx_peak
    else:
        low_before = np.amin(template[:idx_peak])
        idx_low_before = np.argmin(template[:idx_peak])

    if idx_peak == 179:
        low_after = peak
        idx_low_after = idx_peak
    else:
        low_after = np.amin(template[idx_peak:])
        idx_low_after = idx_peak + np.argmin(template[idx_peak:])

    if idx_low_before == 0:
        peak_before = low_before
        idx_peak_before = idx_low_before
    else:
        peak_before = np.amax(template[:idx_low_before])
        idx_peak_before = np.argmax(template[:idx_low_before])

    if idx_low_after == 179:
        peak_after = low_after
        idx_peak_after = idx_low_after
    else:
        peak_after = np.amax(template[idx_low_after:])
        idx_peak_after = idx_low_after + np.argmax(template[idx_low_after:])

    if idx_peak_before == 0:
        low_before_before = peak_before
        idx_low_before_before = idx_peak_before
    else:
        low_before_before = np.amin(template[:idx_peak_before])
        idx_low_before_before = np.argmin(template[:idx_peak_before])

    if idx_peak_after == 179:
        low_after_after = peak_after
        idx_low_after_after = idx_peak_after
    else:
        low_after_after = np.amin(template[idx_peak_after:])
        idx_low_after_after = idx_peak_after + np.argmin(
            template[idx_peak_after:]
        )

    assert idx_low_before_before <= idx_peak_before
    assert idx_peak_before <= idx_low_before
    assert idx_low_before <= idx_peak
    assert idx_peak <= idx_low_after
    assert idx_low_after <= idx_peak_after
    assert idx_peak_after <= idx_low_after_after

    values = [low_before_before, peak_before, low_before, peak, low_after,
              peak_after, low_after_after]
    indices = [idx_low_before_before, idx_peak_before, idx_low_before,
               idx_peak, idx_low_after, idx_peak_after, idx_low_after_after]
    diff_val = [cmb[0] - cmb[1] for cmb in itertools.combinations(values, 2)]
    diff_idx = [cmb[0] - cmb[1] for cmb in itertools.combinations(indices, 2)]

    return np.array(values + indices + diff_val + diff_idx)


def extract_features(ecg_df, save=True, mode='train'):
    '''
    input: raw X_train_df
    '''
    t0 = time()
    # MAIN FEATURES, INITIALIZING #
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
                                'filtered': values.apply(lambda x: x['filtered']),
                                'templates': values.apply(lambda x: x['templates']),
                                'heart_rate': values.apply(lambda x: x['heart_rate'])})

    peaks = ["rpeaks"]
    for i in peaks:
        print(i)
        features_df['val_' + i] = features_df.apply(lambda x: np.array(x['filtered'][x[i]]), axis=1)
        features_df['mean_' + i] = features_df.apply(lambda x: np.mean(x['val_' + i]), axis=1)
        features_df['min_' + i] = features_df.apply(lambda x: np.min(x['val_' + i]), axis=1)
        features_df['max_' + i] = features_df.apply(lambda x: np.max(x['val_' + i]), axis=1)
        features_df['std_' + i] = features_df.apply(lambda x: np.std(x['val_' + i]), axis=1)
        features_df['median_' + i] = features_df.apply(lambda x: np.median(x['val_' + i]), axis=1)
        features_df['abs_eng' + i] = features_df.apply(lambda x: fc.abs_energy(x['val_' + i]), axis=1)
        features_df['lin_trend' + i] = features_df.apply(
            lambda x: fc.linear_trend(x['val_' + i], param=[{'attr': 'slope'}])[0][1], axis=1)
        features_df['approx_entr' + i] = features_df.apply(lambda x: fc.number_cwt_peaks(x['val_' + i], n=50), axis=1)

    features_df['custom_duration'] = features_df.apply(
        lambda x: np.std(np.argmin(x['templates'], axis=0) - np.argmax(x['templates'], axis=0)), axis=1)
    features_df['custom_duration_std'] = features_df.apply(
        lambda x: np.mean(np.argmin(x['templates'], axis=0) - np.argmax(x['templates'], axis=0)), axis=1)
    features_df['custom_duration_min'] = features_df.apply(
        lambda x: np.min(np.argmin(x['templates'], axis=0) - np.argmax(x['templates'], axis=0)), axis=1)
    features_df['custom_duration_max'] = features_df.apply(
        lambda x: np.max(np.argmin(x['templates'], axis=0) - np.argmax(x['templates'], axis=0)), axis=1)

    features_df['custom_argmin_std'] = features_df.apply(lambda x: np.std(np.argmin(x['templates'], axis=0)), axis=1)
    features_df['custom_argmin_min'] = features_df.apply(lambda x: np.min(np.argmin(x['templates'], axis=0)), axis=1)
    features_df['custom_argmin_max'] = features_df.apply(lambda x: np.max(np.argmin(x['templates'], axis=0)), axis=1)
    features_df['custom_argmax'] = features_df.apply(lambda x: np.std(np.argmax(x['templates'], axis=0)), axis=1)
    features_df['custom_argmin_median'] = features_df.apply(
        lambda x: np.median(np.argmin(x['templates'], axis=1)), axis=1)

    features_df['mean_template'] = features_df.apply(lambda x: np.mean(x['templates'], axis=0), axis=1)
    features_df['switch'] = features_df.apply(lambda x: switch(x['mean_template']), axis=1)

    array_df = features_df.apply(lambda x: extract_template_info(x['mean_template']), axis=1)

    for i in range(55):
        features_df[i] = array_df.apply(lambda x: x[i])

    # POWER
    # features_df['power'] = features_df.apply(lambda x: ef(x['filtered'], default_fc_parameters=settings), axis=1)
    features_df['power'] = features_df.apply(lambda x: np.sum(np.square(x['filtered'])) / x['filtered'].shape[0],
                                             axis=1)

    ## CARDIADIC CYCLES
    ## features_df['Cardiadic_Cycles'] = features_df['templates'].apply(lambda x: x)
    # features_df['mean'] = features_df['templates'].apply(lambda x: np.mean(np.mean(x, axis=0)))
    # features_df['mean_median'] = features_df['templates'].apply(lambda x: np.mean(np.median(x, axis=0)))
    # features_df['mean_std'] = features_df['templates'].apply(lambda x: np.mean(np.std(x.astype(np.float), axis=0)))
    # features_df['min_std'] = features_df['templates'].apply(lambda x: np.min(np.std(x.astype(np.float), axis=0)))
    # features_df['max_std'] = features_df['templates'].apply(lambda x: np.max(np.std(x.astype(np.float), axis=0)))
    # Todo replace std with quality signal?

    # HRV FEATURES
    features_names_hvr = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN',
                          'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN',
                          'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF', 'HRV_VLF',
                          'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
                          'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
                          'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
                          'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
                          'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
                          'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn', 'HRV_SampEn']
    features_df['hrv_features'] = features_df.apply(lambda x: nk.hrv(peaks=x['rpeaks'], sampling_rate=300), axis=1)
    for name in features_names_hvr:
        features_df[name] = features_df['hrv_features'].apply(
            lambda x: x[name] if type(x) == pd.core.frame.DataFrame else np.nan)

    # FINALIZE / SAVE
    features_df = features_df.drop(['val_' + s for s in peaks], axis=1)
    features_df = features_df.drop(['heart_rate', 'hrv_features', 'mean_template', 'rpeaks', 'filtered', 'templates'],
                                   axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('final/' + mode + '_' + str(numberOfFeatures) + '.csv', index=False)

    print(mode + " data done in %0.3fs" % (time() - t0))
    return features_df


debug = 0
if debug:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Debug mode activated')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Chose some random samples to load
    rng = np.random.default_rng(seed=1)
    skip = rng.choice(np.arange(1, 5118, step=1), size=5107, replace=False)
    skip = np.arange(30, 5118, step=1).tolist()
    x_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id', skiprows=skip)
else:
    x_test = pd.read_csv('raw/X_test.csv', sep=',', index_col='id')
    x_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id')
    print('Data loaded')

# TRAIN
# X_train_df = pd.read_csv('Project3/raw/X_train.csv')


extract_features(x_train, mode='train')
extract_features(x_test, mode='test')
# # TEST
# X_test_df = pd.read_csv('../raw/mitbih_test.csv')
# extract_features(X_test_df, mode='test')
