import itertools

import numpy as np
import pandas as pd
from biosppy.signals import ecg


# tslearn, tsfresh, sktime

#
# def format(x):
#     x = x.stack().to_frame().reset_index(level=[0, 1])
#     x['level_1'] = x['level_1'].map(lambda x: x.lstrip('x'))
#     x['level_1'] = pd.to_numeric(x['level_1'])
#     x[0] = pd.to_numeric(x[0])
#     return x
#
#
# # Load data
# # Todo try dask
# debug = 1
# if debug:
#     print('Debug mode activated')
#     skip = np.arange(10, 5117, step=1).tolist()
#     X_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id', skiprows=skip)
#     y_train = pd.read_csv('raw/mitbih_train.csv.csv', sep=',', index_col='id', skiprows=skip)
# else:
#     X_test = pd.read_csv('raw/mitbih_test.csv', sep=',', index_col='id')
#     X_train = pd.read_csv('raw/X_train.csv', sep=',', index_col='id')
#     y_train = pd.read_csv('raw/mitbih_train.csv.csv', sep=',', index_col='id')
#
# # Todo Nan currently dropped
# X_train = tsfresh.extract_features(X_train.stack(), column_id='id', n_jobs=4)
# X_train.to_pickle('feat/X_train_feat.zip')
# X_test = tsfresh.extract_features(X_test.stack(), n_jobs=-1)
# X_test.to_pickle('feat/X_test_feat.zip')
# impute(X_train)
# features_filtered = select_features(X_train, y_train)

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


def features_from_raw_data(data):
    features = []
    cnt = 0
    for series in data:
        print(str(cnt))
        cnt += 1
        # print (series)
        # Todo: Check this conversion!!!
        series = series[np.logical_not(np.isnan(series))]
        # print (series)
        # print (series.shape)
        # series = series[np.bitwise.isnan(series)]
        out = ecg.ecg(series, sampling_rate=300, show=False)

        filtered = np.array(out[1])  # Filtered ECG signal. (ARRAY)
        rpeaks = np.array(out[2])  # R-peak location indices (ARRAY)
        templates = np.array(out[4])  # Extracted heartbeat templates. (ARRAY OF ARRAY)
        heart_rate = np.array(out[6])  # Instantaneous heart rate (bpm). (ARRAY)
        # print(filtered)
        # print(rpeaks)
        # print(templates.shape)
        # print(heart_rate)
        series_features = []  # ARRAY [series info..., filtered info..., heartrate info..., peak dist info..., template info...]

        # #extrahiere aus jedem 1d array featurevector
        # for info in [series, filtered, heart_rate, rpeaks[1:] - rpeaks[:-1]]: #letzte ist abst., zw. peaks
        #     if info.shape[0] == 0:
        #         series_features.extend([0, 0, 0, 0, 0, 0])
        #     else:
        #         series_features.extend(
        #             [
        #                 np.mean(info), np.var(info), np.median(info),
        #                 np.amin(info), np.amax(info), np.amax(info) - np.amin(info)
        #             ]
        #         )
        # #array of arrays, immer features au 0,1,2,... position (ueber collom)
        # series_features.extend(np.mean(templates, axis=0).tolist())
        # series_features.extend(np.var(templates, axis=0).tolist())
        # series_features.extend(np.median(templates, axis=0).tolist())
        # series_features.extend(np.amin(templates, axis=0).tolist())
        # series_features.extend(np.amax(templates, axis=0).tolist())

        template_values = []

        for i in range(templates.shape[0]):
            template_values.append(extract_template_info(templates[i, :]))

        template_values = np.vstack(template_values)

        series_features.extend(np.mean(template_values, axis=0).tolist())
        series_features.extend(np.var(template_values, axis=0).tolist())
        series_features.extend(np.median(template_values, axis=0).tolist())
        series_features.extend(np.amin(template_values, axis=0).tolist())
        series_features.extend(np.amax(template_values, axis=0).tolist())

        mean_template = np.mean(templates, axis=0)
        series_features.extend(extract_template_info(mean_template).tolist())

        features.append(np.array(series_features))
    return np.vstack(features)


#X_train = pd.read_csv('raw/X_train.csv').drop(['id'], axis=1, dtype='Int64').values
X_test = pd.read_csv('raw/X_test.csv', dtype='Int64').drop(['id'], axis=1).values
#Train_features = features_from_raw_data(X_train)
#np.savetxt('feat/features_train_temp_only.csv', Train_features, delimiter=',', fmt='%s')
Test_features = features_from_raw_data(X_test)
np.savetxt('feat/features_test_temp_only.csv', Test_features, delimiter=',', fmt='%s')
