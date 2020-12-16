import pandas as pd
import numpy as np


def combine_5_epochs(df):
    '''
    conbines features from the last 2 epochs and next 2 epochs,
    to be estimated is the epoch in the middle
    '''
    features = list(df.columns)
    features_2_ = [x + '_2_' for x in features]
    features_1_ = [x + '_1_' for x in features]
    features__1 = [x + '__1' for x in features]
    features__2 = [x + '__2' for x in features]

    all_features = [x + '_2_' for x in features]
    all_features.extend(features_1_)
    all_features.extend(features)
    all_features.extend(features__1)
    all_features.extend(features__2)

    new_df = pd.DataFrame(columns=all_features)
    for row in df.iterrows():
        index = row[0]
        if index % 21600 == 0:
            epoch_2_ = dict(zip(features_2_, list(row[1].values)))
            epoch_1_ = dict(zip(features_1_, list(row[1].values)))
            epoch_mid = dict(row[1])
            epoch__1 = dict(zip(features__1, list(df.iloc[index + 1].values)))
            epoch__2 = dict(zip(features__2, list(df.iloc[index + 2].values)))
        elif index % 21600 == 1:
            epoch_2_ = dict(zip(features_2_, list(df.iloc[index - 1].values)))
            epoch_1_ = dict(zip(features_1_, list(df.iloc[index - 1].values)))
            epoch_mid = dict(row[1])
            epoch__1 = dict(zip(features__1, list(df.iloc[index + 1].values)))
            epoch__2 = dict(zip(features__2, list(df.iloc[index + 2].values)))
        elif index % 21600 == 21599:
            epoch_2_ = dict(zip(features_2_, list(df.iloc[index - 2].values)))
            epoch_1_ = dict(zip(features_1_, list(df.iloc[index - 1].values)))
            epoch_mid = dict(row[1])
            epoch__1 = dict(zip(features__1, list(row[1].values)))
            epoch__2 = dict(zip(features__2, list(row[1].values)))
        elif index % 21600 == 21598:
            epoch_2_ = dict(zip(features_2_, list(df.iloc[index - 2].values)))
            epoch_1_ = dict(zip(features_1_, list(df.iloc[index - 1].values)))
            epoch_mid = dict(row[1])
            epoch__1 = dict(zip(features__1, list(df.iloc[index + 1].values)))
            epoch__2 = dict(zip(features__2, list(df.iloc[index + 1].values)))
        else:
            epoch_2_ = dict(zip(features_2_, list(df.iloc[index - 2].values)))
            epoch_1_ = dict(zip(features_1_, list(df.iloc[index - 1].values)))
            epoch_mid = dict(row[1])
            epoch__1 = dict(zip(features__1, list(df.iloc[index + 1].values)))
            epoch__2 = dict(zip(features__2, list(df.iloc[index + 2].values)))
        final_feature = dict(epoch_2_, **epoch_1_)
        final_feature.update(epoch_mid)
        final_feature.update(epoch__1)
        final_feature.update(epoch__2)
        new_df = new_df.append(final_feature, ignore_index=True)

    return new_df