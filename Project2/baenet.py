
from pprint import pprint
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, f_classif, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram


def lof(x, y):
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def isof(x, y):
    model = IsolationForest()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]

# Load data
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv', index_col='id')
X_train = pd.read_csv('raw/X_train.csv', index_col='id')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

# Reduce Data for debugging
evals = 100
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.9)
    del a, b
    evals = 3
    print('Debug mode on')

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
    ('outlier', 'passthrough'),
    ('scale', RobustScaler()),
    ('model', 'passthrough')
])
# Specify parameters to be searched over
svc_search = {
        'outlier': Categorical([FunctionSampler(func=isof), FunctionSampler(func=lof)]),  # , FunctionSampler(func=lof)
        'model': [SVC(class_weight='balanced', decision_function_shape='ovo')]
    }

# Haperparameter opt
opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(sgd_search,evals), (nusvc_search, evals), (svc_search, evals)],
    cv=3,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=3
)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe.steps])
print("parameters:")
pprint(svc_search)
t0 = time()
opt.fit(X_train, y_train.values.ravel())
print("Done in %0.3fs" % (time() - t0))
print()

# Evaluate Results
print('Best', opt.scoring, 'score: ', opt.best_score_)
print("Best parameters set:", opt.best_params_)
print()
print("Grid scores on development set:")
print()
means = opt.cv_results_['mean_test_score']
stds = opt.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, opt.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# Predict for test set
y_test = opt.best_estimator_.predict(X_test)
print()

# Save prediction
y_test = pd.DataFrame(y_test)
y_test.to_csv('prediction.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
