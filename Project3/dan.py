import numpy as np
import pandas as pd
import neurokit2 as nk


def extract(x):
    ecg_analysze_feat_col = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
           'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
           'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
           'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
           'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
           'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS',
           'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d',
           'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
           'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn',
           'HRV_SampEn']
    features = pd.DataFrame(columns=ecg_analysze_feat_col)

    for i, row in x.iterrows():
        if i % 10 == 0:
            print(i)
        # Todo: add quality analysis from signals

        # Get the current signal adn trim nans
        currentPatient = row.dropna().values

        # Preprocess ECG signal
        # info: A dictionary containing the samples at which the R-peaks occur, accessible with the key
        # signals: various processed ecg signals. The peaks coloumn mark with 1 when a peak occurs
        signals, info = nk.ecg.ecg_process(currentPatient, sampling_rate=300)
        ecg_analyze_features = nk.ecg.ecg_analyze(signals, sampling_rate=300)

        # Peaks: Aggregate time series: mean, max, min, median, std
        # peaks = ['ECG_R_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks', 'ECG_P_Onsets', 'ECG_T_Offsets']
        # for j in peaks:
        #     peak_amp = signals['ECG_Clean'][signals['j'] == 1].values
        #     mean = np.mean(peak_amp)
        #     min = np.min(peak_amp)
        #     max = np.max(peak_amp)
        #     std = np.std(peak_amp)
        #     median = np.median(peak_amp)

        # ECG_Rate: Aggregate time series: mean, max, min, median, std
        # heart_rate = signals['ECG_Clean'].values
        # mean_hr = np.mean(heart_rate)
        # max_hr = np.min(heart_rate)
        # min_hr = np.max(heart_rate)
        # median_hr = np.std(heart_rate)
        # std_hr = np.median(heart_rate)

        # Time series aggregation on mean template (one pulse R_peak to R_peak
        # epochs = nk.ecg_segment(signals['ECG_Clean'].values, rpeaks=info['ECG_R_Peaks'], sampling_rate=300, show=False)
        # The filtered ECG signal
        # filtered = signals['ECG_Clean'].values
        # add power
        # power = np.sum(np.square(filtered)) / filtered.shape[0]
        # features = np.append(features, power.reshape(1, -1), axis=1)


        # # Addition of frequency features
        # hvr_freq_features = frequency_analysis.get_frequency_features(templates)
        # # add all frequency hvr to array features
        # for feature in features_names_freqhvr:
        #     features = np.append(features, hvr_freq_features[feature])

        features = features.append(ecg_analyze_features)
    return features

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
    print('Data loaded from memory')

train_feat = extract(x_train)
train_feat.to_csv('feat2/train_feat')

#test_feat = extract(x_test)
#test_feat.to_csv('feat2/test_feat')
