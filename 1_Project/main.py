import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler


# Import data
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv')
X_train = pd.read_csv('raw/X_train.csv')
y_train = pd.read_csv('raw/y_train.csv')

# Inspect data
percent_missing = X_test.isnull().sum() * 100 / len(X_test)
missing_value_df = pd.DataFrame({'column_name': X_test.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
# X_train is missing 3%-10% of the values

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('scale', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
])

param_grid = [
    {
        'scale': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]