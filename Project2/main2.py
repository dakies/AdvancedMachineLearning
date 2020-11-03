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
from sklearn.svm import SVR, SVC, OneClassSVM
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
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv', index_col='id')
X_train = pd.read_csv('raw/X_train.csv', index_col='id')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

# Reduce Data for debugging
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.999)
    del a, b
    print('Debug mode on')

param_grid_xgb = [
    {
        'estimator__min_child_weight': [1, 5, 10],
        'estimator__gamma': [0, 1, 2],
        'estimator__subsample': [0.6, 0.8, 1.0],
        'estimator__colsample_bytree': [0.6, 0.8, 1.0],
        'estimator__max_depth': [3, 5]
    }
]

# Gridsearch
clf = OneVsRestClassifier(estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                colsample_bynode=1, colsample_bytree=1, gamma=0,
                                importance_type='gain', learning_rate=0.1, max_delta_step=0,
                                max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                                n_jobs=1, nthread=None, random_state=0,
                                reg_alpha=0, reg_lambda=1, seed=None,
                                silent=None, subsample=1, verbosity=1))
search = GridSearchCV(clf, param_grid=param_grid_xgb, n_jobs=-1, scoring='balanced_accuracy', cv=5, verbose=10)
print("Performing grid search...")
#print("pipeline:", [name for name, _ in pipe_xgb.steps])
print("parameters:")
pprint(param_grid_xgb)
t0 = time()

# preprocessing #
thresh = VarianceThreshold(threshold=0) # constant columns
X_train = thresh.fit_transform(X_train)
X_test = thresh.transform(X_test)

X_train, y_train = isof(X_train, y_train.values.ravel()) # outlier

sampler = SMOTEENN() # downsample
X_train, y_train = sampler.fit_resample(X_train, y_train)

selection = GenericUnivariateSelect(mode='fwe', param=0.1) # feature selection
X_train = selection.fit_transform(X_train, y_train)
X_test = selection.transform(X_test)

scale = StandardScaler() # scaling
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# instance weights #
#X_train, y_train, weights = instance_weights(X_train, y_train)

search.fit(X_train, y_train)
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
y_test.to_csv('prediction_xgb.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
