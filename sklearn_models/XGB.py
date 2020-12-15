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
from sklearn.feature_selection import SelectKBest, f_classif


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
X_test = pd.read_csv('./features/features_test_43.csv')
X_train = pd.read_csv('./features/features_train_43.csv')
y_train = pd.read_csv('../raw/y_train.csv', index_col='id')

###
train_all = pd.concat([X_train, y_train], axis=1).dropna(axis=0).reset_index(drop=True)
X_train = train_all.drop('y', axis=1)
y_train = pd.DataFrame(train_all['y'])
X_test = X_test.fillna(method='bfill', axis=0)
###

# Reduce Data for debugging
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.999)
    del a, b
    print('Debug mode on')

# Create pipeline

# pipe_xgb = Pipeline([
#     # the scale stage is populated by the param_grid
#     ('remove const cloumns', VarianceThreshold(threshold=0)),
#     ('outlier', 'passthrough'),
#     ('sample', 'passthrough'),
#     ('selection', 'passthrough'),
#     ('scale', StandardScaler()),
#     #('instance_weights', 'passthrough'),
#     ('estimation', 'passthrough')
# ])

pipe_xgb = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
    ('outlier', 'passthrough'),
    ('sample', 'passthrough'),
    ('selection',  SelectKBest(f_classif)),
    ('scale', StandardScaler()),
    #('instance_weights', 'passthrough'),
    ('estimation', 'passthrough')
])

# Specify parameters to be searched over

# param_grid_xgb = [
#     {
#         'outlier': [FunctionSampler(func=isof), None],# FunctionSampler(func=oneclasssvm)],  #, FunctionSampler(func=lof)
#         'sample': [SMOTEENN(), RandomOverSampler(), None],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
#         'selection': [GenericUnivariateSelect(mode='fdr', param=0.2), SelectKBest(f_classif, k = 40)],
#         #'selection__mode': ['fwe', 'fdr'], #'fpr', 'fdr',
#         #'selection__param': [0.1, 0.2],
#         #'instance_weights': [FunctionSampler(func=instance_weights)],
#         'estimation': [OneVsRestClassifier(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                                 colsample_bynode=1, colsample_bytree=1, gamma=0,
#                                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
#                                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#                                 n_jobs=1, nthread=None, random_state=0,
#                                 reg_alpha=0, reg_lambda=1, seed=1111,
#                                 silent=None, subsample=1, verbosity=1))],
#         'estimation__estimator__min_child_weight': [7,9,11],
#         'estimation__estimator__gamma': [0, 1, 2],
#         'estimation__estimator__n_estimators': [200,400,600],
#         'estimation__estimator__subsample': [0.6, 0.8, 1.0],
#         'estimation__estimator__colsample_bytree': [0.6, 0.8, 1.0],
#         'estimation__estimator__max_depth': [5,7,10]
#     }
# ]

param_grid_xgb = [
    {
        'outlier': [FunctionSampler(func=isof), None],# FunctionSampler(func=oneclasssvm)],  #, FunctionSampler(func=lof)
        'sample': [None],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        #'selection': [GenericUnivariateSelect(mode='fdr', param=0.2), SelectKBest(f_classif, k = 40), None],
        'selection__k': [35,37,40],
        #'selection__mode': ['fwe', 'fdr'], #'fpr', 'fdr',
        #'selection__param': [0.1, 0.2],
        #'instance_weights': [FunctionSampler(func=instance_weights)],
        'estimation': [XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, objective= 'binary:logistic',
                                colsample_bynode=1, colsample_bytree=1, gamma=0, num_class = 4,
                                importance_type='gain', learning_rate=0.05, max_delta_step=0,
                                max_depth=3, min_child_weight=1, missing=None,
                                n_jobs=1, nthread=None, random_state=0,
                                reg_alpha=0, reg_lambda=1, seed=1111,
                                silent=None, subsample=1, verbosity=1)],
        'estimation__min_child_weight': [7,9,11],
        'estimation__gamma': [0, 1, 2],
        'estimation__n_estimators': [200,400,600],
        'estimation__subsample': [0.6, 0.8, 1.0],
        'estimation__colsample_bytree': [0.6, 0.8, 1.0],
        'estimation__max_depth': [5,7,10]
    }
]

# Gridsearch
search = GridSearchCV(pipe_xgb, param_grid=param_grid_xgb, n_jobs=-1, scoring='f1_micro', cv=3, verbose=10)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe_xgb.steps])
print("parameters:")
pprint(param_grid_xgb)
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
y_test.to_csv('prediction_xgb_43_last.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction_xgb_43_last.csv')


