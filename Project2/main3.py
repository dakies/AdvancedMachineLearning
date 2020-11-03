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
from sklearn.neural_network import MLPClassifier


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

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
    ('outlier', 'passthrough'),
    ('sample', 'passthrough'),
    ('selection', GenericUnivariateSelect()),
    ('scale', StandardScaler()),
    ('estimation', 'passthrough')
])

pipe_nn = Pipeline([
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
param_grid = [
    # Feature selection: SelectKBest
    # {
    #     'outlier': [FunctionSampler(func=isof), FunctionSampler(func=lof)],  #
    #     'sample': [RandomOverSampler(), SMOTE(), ADASYN()],  #
    #     'selection': [SelectFromModel(ExtraTreesClassifier(n_estimators=50))],
    #     'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
    #     'estimation__kernel': ['rbf'],
    #     'estimation__C': [0.001, 0.01, 0.1, 10]
    # },
    {
        'outlier': [FunctionSampler(func=isof)],  #, FunctionSampler(func=lof)
        'sample': [RandomUnderSampler()],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        'selection__mode': ['fwe'], #'fpr', 'fdr',
        'selection__param': [0.1],
        'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'estimation__kernel': ['rbf'],
        'estimation__C': np.logspace(0, 2, num=4),
        'estimation__gamma':  ['auto']
    } #, {
    #     'outlier': [FunctionSampler(func=isof), FunctionSampler(func=lof)],  #
    #     'sample': [RandomOverSampler(), RandomUnderSampler(), SMOTEENN(), SMOTETomek()],  #
    #     'selection__mode': ['percentile'],
    #     'selection': [GenericUnivariateSelect(f_classif)],
    #     'selection__param': [10, 30, 50],
    #     'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
    #     'estimation__kernel': ['rbf'],
    #     'estimation__C': [0.001, 0.01, 0.1, 10]
    # } # ,
    # {
    #     'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
    #     'sample': [RandomOverSampler()],  # , SMOTE(), ADASYN()
    #     'selection__mode': ['fpr', 'fdr', 'fwe'],
    #     'selection__param': [0.00005, 0.0001, 0.001, 1],
    #     'selection__score_func': [f_classif],
    #     'estimation': [GaussianNB()]
    # }, {
    #     'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
    #     'sample': [RandomOverSampler()],  # , SMOTE(), ADASYN()
    #     'selection__mode': ['fpr', 'fdr', 'fwe'],
    #     'selection__param': [0.00005, 0.0001, 0.001, 1],
    #     'selection__score_func': [f_classif],
    #     'estimation': [KNeighborsClassifier()],
    #     'estimation__weights': ['uniform', 'distance'],
    #     'estimation__leaf_size': [10, 30, 40],
    #     'estimation__n_neighbors': [3, 5, 10],
    #
    # }, {
    #     'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
    #     'sample': [RandomOverSampler()],  # , SMOTE(), ADASYN()
    #     'selection__mode': ['fpr', 'fdr', 'fwe'],
    #     'selection__param': [0.00005, 0.0001, 0.001, 1],
    #     'selection__score_func': [f_classif],
    #     'estimation': [RandomForestClassifier()]
    # }

]

param_grid_nn = [
    {
        'outlier': [FunctionSampler(func=isof)],# FunctionSampler(func=oneclasssvm)],  #, FunctionSampler(func=lof)
        'sample': [RandomUnderSampler(), SMOTEENN()],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        'selection__mode': ['fwe'], #'fpr', 'fdr',
        'selection__param': [0.1],
        #'instance_weights': [FunctionSampler(func=instance_weights)],
        'estimation': [OneVsRestClassifier(estimator = MLPClassifier(activation='relu', n_iter_no_change=20,
                                                                     early_stopping=True, max_iter=300))],
        'estimation__estimator__hidden_layer_sizes': [(i,) for i in range(10,60,10)] +
                                                     [(i,i) for i in range(10,60,10)] +
                                                     [(i,i,i) for i in range(10,60,10)],
        'estimation__estimator__alpha': [0.1, 1, 10, 100]
    }
]

# Gridsearch
search = GridSearchCV(pipe_nn, param_grid=param_grid_nn, n_jobs=-1, scoring='balanced_accuracy', cv=2, verbose=10)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe_nn.steps])
print("parameters:")
pprint(param_grid_nn)
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
y_test.to_csv('prediction_nn.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
