import biosppy.signals.ecg as ecg
import neurokit2 as nk2
import numpy as np
import pandas as pd
from neurokit2.ecg.ecg_clean import ecg_clean
from neurokit2.ecg.ecg_delineate import ecg_delineate
from neurokit2.ecg.ecg_peaks import ecg_peaks
from neurokit2.ecg.ecg_phase import ecg_phase
from neurokit2.ecg.ecg_quality import ecg_quality
from neurokit2.signal import signal_rate, signal_sanitize


# import frequency_analysis

def my_process(ecg_signal, sampling_rate=300, method="neurokit"):
    # -*- coding: utf-8 -*-

    """Process an ECG signal.
    Convenience function that automatically processes an ECG signal.
    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Defaults to "neurokit".
    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the `ecg_signal` containing the following columns:
        - *"ECG_Raw"*: the raw signal.
        - *"ECG_Clean"*: the cleaned signal.
        - *"ECG_R_Peaks"*: the R-peaks marked as "1" in a list of zeros.
        - *"ECG_Rate"*: heart rate interpolated between R-peaks.
        - *"ECG_P_Peaks"*: the P-peaks marked as "1" in a list of zeros
        - *"ECG_Q_Peaks"*: the Q-peaks marked as "1" in a list of zeros .
        - *"ECG_S_Peaks"*: the S-peaks marked as "1" in a list of zeros.
        - *"ECG_T_Peaks"*: the T-peaks marked as "1" in a list of zeros.
        - *"ECG_P_Onsets"*: the P-onsets marked as "1" in a list of zeros.
        - *"ECG_P_Offsets"*: the P-offsets marked as "1" in a list of zeros
                            (only when method in `ecg_delineate` is wavelet).
        - *"ECG_T_Onsets"*: the T-onsets marked as "1" in a list of zeros
                            (only when method in `ecg_delineate` is wavelet).
        - *"ECG_T_Offsets"*: the T-offsets marked as "1" in a list of zeros.
        - *"ECG_R_Onsets"*: the R-onsets marked as "1" in a list of zeros
                            (only when method in `ecg_delineate` is wavelet).
        - *"ECG_R_Offsets"*: the R-offsets marked as "1" in a list of zeros
                            (only when method in `ecg_delineate` is wavelet).
        - *"ECG_Phase_Atrial"*: cardiac phase, marked by "1" for systole
          and "0" for diastole.
        - *"ECG_Phase_Ventricular"*: cardiac phase, marked by "1" for systole and "0" for diastole.
        - *"ECG_Atrial_PhaseCompletion"*: cardiac phase (atrial) completion, expressed in percentage
          (from 0 to 1), representing the stage of the current cardiac phase.
        - *"ECG_Ventricular_PhaseCompletion"*: cardiac phase (ventricular) completion, expressed in
          percentage (from 0 to 1), representing the stage of the current cardiac phase.
    info : dict
        A dictionary containing the samples at which the R-peaks occur, accessible with the key
        "ECG_Peaks".
    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_plot, signal_rate, signal_fixpeaks
    Examples
    --------
    <Figure ...>
    """
    # Sanitize input
    ecg_signal = signal_sanitize(ecg_signal)

    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method)
    # R-peaks
    instant_peaks, rpeaks, = ecg_peaks(
        ecg_cleaned=ecg_cleaned, sampling_rate=sampling_rate, method=method, correct_artifacts=True
    )

    rate = signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    quality = ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate)

    signals = pd.DataFrame(
        {"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})

    # Additional info of the ecg signal
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, method="dwt"
    )

    cardiac_phase = ecg_phase(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)

    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    info = rpeaks
    return signals, info


def ecg_process_custom(ecg_signal, sampling_rate=1000, method='neurokit'):
    try:
        output = my_process(ecg_signal, sampling_rate, method)
    except:
        output = np.nan
    return output


def extract_features(ecg_df, save=True, mode='train'):  ## feature amount = 42
    '''
    input: raw X_train_df
    '''

    features_names_timehvr = ['sdNN', 'meanNN', 'CVSD', 'cvNN', 'RMSSD', 'medianNN',
                              'madNN', 'mcvNN', 'pNN50', 'pNN20']
    additional_peaks = ["ECG_P_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_P_Offsets", "ECG_T_Onsets",
                        "ECG_T_Offsets",
                        "ECG_R_Onsets", "ECG_R_Offsets"]

    # MAIN FEATURES, INITIALIZING #
    print('-- EXTRACTING MAIN FEATURES --')
    values = ecg_df.apply(lambda x: ecg.ecg(x.dropna(), sampling_rate=300, show=False), axis=1)
    features_df = pd.DataFrame({'rpeaks': values.apply(lambda x: x['rpeaks']),
                                'filtered': values.apply(lambda x: x['filtered']),
                                'templates': values.apply(lambda x: x['templates'])})
    print('-- VALUES EXTRACTION DONE --')
    # values_nk2 = ecg_df.apply(lambda x: ecg_process_custom(x.dropna(), sampling_rate=300)[0], axis=1)
    values_nk2 = ecg_df.apply(lambda x: ecg_process_custom(x.dropna(), sampling_rate=300), axis=1)
    values_nk2 = values_nk2.apply(lambda x: x[0] if x == x else np.nan)
    for v in additional_peaks:
        peak_name = k[0]
        features_df[k] = values_nk2.apply(lambda x: np.array(x[v]) if type(
            x) == pd.core.frame.DataFrame else np.nan)  # problem in this line change x==x
        features_df[k] = features_df[k].apply(lambda x: np.where(x == 1)[0] if type(x) == np.ndarray else np.nan)
        print('----->' + peak_name + ' done.')
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
    for k in additional_peaks:
        features_df[k] = features_df.apply(lambda x: x['filtered'][x[k]] if type(x[k]) == np.ndarray else np.nan,
                                           axis=1)

        peak_name = k[0]
        features_df['mean_' + peak_name + 'values'] = features_df.apply(
            lambda x: np.mean(x[k]) if (type(x[k]) == np.ndarray and len(x[k]) != 0) else np.nan, axis=1)
        features_df['min_' + peak_name + 'values'] = features_df.apply(
            lambda x: np.min(x[k]) if (type(x[k]) == np.ndarray and len(x[k]) != 0) else np.nan, axis=1)
        features_df['max_' + peak_name + 'values'] = features_df.apply(
            lambda x: np.max(x[k]) if (type(x[k]) == np.ndarray and len(x[k]) != 0) else np.nan, axis=1)
        features_df['std_' + peak_name + 'values'] = features_df.apply(
            lambda x: np.std(x[k]) if (type(x[k]) == np.ndarray and len(x[k]) != 0) else np.nan, axis=1)
        features_df['median_' + peak_name + 'values'] = features_df.apply(
            lambda x: np.median(x[k]) if (type(x[k]) == np.ndarray and len(x[k]) != 0) else np.nan, axis=1)
        print('----->' + peak_name + ' done.')
    print('-- OTHER PEEKS EXTRACTION DONE --')

    # POWER
    features_df['power'] = features_df['filtered'].apply(lambda x: np.sum(np.square(x)) / x.shape[0])

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
    features_df['hrv_features'] = features_df.apply(lambda x: nk2.hrv(peaks=x['info'], sampling_rate=300), axis=1)
    for name in features_names_hvr:
        features_df[name] = features_df['hrv_features'].apply(lambda x: x[name])
    print('-- HRV EXTRACTION DONE --')

    # FINALIZE / SAVE
    features_df = features_df.drop(['rpeaks', 'filtered', 'templates', 'R_peaks', 'hrv_features'], axis=1)
    features_df = features_df.drop(list(additional_peaks), axis=1)
    numberOfFeatures = len(features_df.columns)

    if save:
        features_df.to_csv('./features/features_' + mode + '_' + str(numberOfFeatures) + '.csv', index=False)
        print('-- FEATURES SAVED --')

    return features_df


# TRAIN
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
X_train_df = x_train
extract_features(X_train_df)
extract_features(x_test, mode='test')

# # TEST
# X_test_df = pd.read_csv('../raw/X_test.csv')
# extract_features(X_test_df, mode='test')
