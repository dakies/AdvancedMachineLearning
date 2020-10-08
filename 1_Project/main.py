from pprint import pprint
from time import time
import logging
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression


from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler


def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    print('Initiating Outlier detection')
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(X)
    print('Outliers removed', X[y_pred == -1].shape[0])
    return X[y_pred == 1], y[y_pred == 1]


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Import data
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv', index_col='id')
X_train = pd.read_csv('raw/X_train.csv', index_col='id')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

# Reduce Data for debugging
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.99)
    del a, b
    print('Debug mode on')

# Inspect data
percent_missing = X_test.isnull().sum() * 100 / len(X_test)
missing_value_df = pd.DataFrame({'column_name': X_test.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
# X_train is missing 3%-10% of the values

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('impute', SimpleImputer()),
    ('outlier',FunctionSampler(func=outlier_rejection)),
    ('scale', 'passthrough'),
    ('selection', SelectKBest(f_regression)),
    ('estimation', SVR())
])

# Specify parameters to be searched over
param_grid = [
    {
        'scale': [RobustScaler()],  # , StandardScaler(), Normalizer()
        'impute__strategy': ['mean', 'median'],
        'selection__k':[90, 100],
        'estimation__kernel': ['rbf'],
        'estimation__C': [10, 50, 100, 500, 1000]

    }
]

# Gridsearch
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='r2')
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe.steps])
print("parameters:")
pprint(param_grid)
t0 = time()
search.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()


# Evaluate Results
print('Best R2 score: ', search.best_score_)
print("Best parameters set:", search.best_params_)
print()
print("Grid scores on development set:")
print()
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


# Train best estimator on whole data
best = search.best_estimator_.fit(X_train, y_train)
# Predict for test set
y_test = best.predict(X_test)
print()


# Save prediction
y_test = pd.DataFrame(y_test)
y_test.to_csv('prediction.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
