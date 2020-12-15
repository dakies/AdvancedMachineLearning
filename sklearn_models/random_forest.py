import logging
from pprint import pprint
from time import time
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn import FunctionSampler
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, f_classif, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR, SVC, OneClassSVM, NuSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier


def lof(x, y):
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def isof(x, y):
    model = IsolationForest()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]

def oneclasssvm(x, y):
    model = OneClassSVM()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]

# ---------------------

def instance_weights(x, y):
    classes, counts = np.unique(y, return_counts=True)
    d = {k:v for k,v in zip(classes, counts)}
    least_count = counts.min()
    weights = np.array([least_count/d[i] for i in y])
    return x,y,weights


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Load data
sample = pd.read_csv('../raw/sample.csv')
X_test = pd.read_csv('./features/features_test_27.csv')
X_train = pd.read_csv('./features/features_train_27.csv')
y_train = pd.read_csv('../raw/y_train.csv', index_col='id')

# Reduce Data for debugging
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.999)
    del a, b
    print('Debug mode on')

# Create pipeline
pipe_rf = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
    ('outlier', 'passthrough'),
    ('sample', 'passthrough'),
    ('selection', GenericUnivariateSelect()),
    ('scale', StandardScaler()),
    #('instance_weights', 'passthrough'),
    ('estimation', 'passthrough')
])

# Specify parameters to be searched over
param_grid_rf = [
    {
        'outlier': [FunctionSampler(func=isof)],# FunctionSampler(func=oneclasssvm)],  #, FunctionSampler(func=lof)
        'sample': [RandomOverSampler(), SMOTEENN(), None],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        'selection__mode': ['fwe', 'fdr'], #'fpr', 'fdr',
        'selection__param': [0.1],
        'estimation': [OneVsRestClassifier(RandomForestClassifier())],
        'estimation__estimator__bootstrap': [True],
        'estimation__estimator__max_depth': [80, 100, 120],
        'estimation__estimator__max_features': [2, 3],
        'estimation__estimator__min_samples_leaf': [3, 4, 5],
        'estimation__estimator__min_samples_split': [8, 10, 12],
        'estimation__estimator__n_estimators': [100, 200, 300, 500]
    }
]

# Gridsearch
search = GridSearchCV(pipe_rf, param_grid=param_grid_rf, n_jobs=-1, scoring='f1_micro', cv=3, verbose=10)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe_rf.steps])
print("parameters:")
pprint(param_grid_rf)
t0 = time()
search.fit(X_train, y_train.values.ravel())
print("Done in %0.3fs" % (time() - t0))
print()

# Evaluate Results
print('Best', search.scoring, 'score: ', search.best_score_)
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


# Predict for test set
y_test = search.best_estimator_.predict(X_test)
print()

# Save prediction
y_test = pd.DataFrame(y_test)
y_test.to_csv('prediction_rf.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')


