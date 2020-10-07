import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import xgboost as xgb

# Import data
from sklearn.svm import SVR

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

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('scale', 'passthrough'),
    ('impute', SimpleImputer()),
    ('feature_selection', SelectFromModel(Lasso())),
    ('estimation', 'passthrough')
])

# Specify parameters to be searched over
param_grid = [
    {
        'scale': [Normalizer(), StandardScaler(), RobustScaler()],
        'impute__strategy': ['mean', 'median', 'most_frequent'],
        'feature_selection__estimator__alpha': [0.1, 1, 10],
        'estimation': [Ridge(alpha=0.1),Ridge(alpha=1), Ridge(alpha=10), SVR()]
    }, {
        'scale': [Normalizer(), StandardScaler(), RobustScaler()],
        'impute__strategy': ['mean', 'median', 'most_frequent'],
        'estimation': [xgb.XGBRegressor()]
    }
]

# Gridsearch
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='r2')
print("Starting Gridsearch")
search.fit(X_train, y_train)

# Evaluate Results
print("Best parameters set found on development set:")
print()
print('R2 score: ', search.best_score_)
print(search.best_params_)
print()
print("Grid scores on development set:")
print()
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# Predict for test set
y_test = search.predict(X_test)
print()
y_test.to_csv('prediction.zip')
print('Results saved as prediction.zip')
# Save prediction

