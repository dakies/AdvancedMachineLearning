import logging
from pprint import pprint
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
<<<<<<< HEAD
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from imblearn import FunctionSampler
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
=======
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, SelectFromModel, RFE
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC
from imblearn import FunctionSampler
>>>>>>> master
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
<<<<<<< HEAD
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR, SVC, NuSVC

=======
from sklearn.tree import DecisionTreeClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
>>>>>>> master

 #Submission

def lof(x, y):
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def isof(x, y):
    model = IsolationForest()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Load data
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv', index_col='id')
X_train = pd.read_csv('raw/X_train.csv', index_col='id')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')


# Reduce Data for debugging
evals = 500
if 0:
<<<<<<< HEAD
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.99)
=======
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.9)
>>>>>>> master
    del a, b
    evals = 3
    print('Debug mode on')

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
    ('outlier', 'passthrough'),
    ('sample', 'passthrough'),
    ('selection', GenericUnivariateSelect()),
    ('scale', RobustScaler()),
    ('model', 'passthrough')
])

# Specify parameters to be searched over
<<<<<<< HEAD
param_grid = [
    # {
    #     'outlier': [FunctionSampler(func=isof)],  #, FunctionSampler(func=lof)
    #     'sample': [RandomUnderSampler()],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
    #     'selection__mode': ['fwe'], #'fpr', 'fdr',
    #     'selection__param': [0.1],
    #     'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
    #     'estimation__kernel': ['rbf'],
    #     'estimation__C': np.logspace(0, 2, num=4),
    #     'estimation__gamma':  ['auto']
    # } #,
    {
        'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
        'sample': [None, RandomUnderSampler(), RandomOverSampler(), SMOTEENN(), SMOTETomek(), SMOTE(), ADASYN()],  #
        'selection': [None],
        'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'estimation__kernel': ['rbf'],
        'estimation__C': np.linspace(1.4, 1.8, num=5),
        'estimation__gamma': ['auto']
    },{
        'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
        'sample': [None, RandomUnderSampler(), RandomOverSampler(), SMOTEENN(), SMOTETomek(), SMOTE(), ADASYN()],  #
        'selection__mode': ['fwe'],
        'selection__param': [0.1],
        'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'estimation__kernel': ['rbf'],
        'estimation__C': np.linspace(1.4, 1.8, num=5),
        'estimation__gamma': ['auto']
    },
    #  percentile Worse than no feature selection
    # {
    #     'outlier': [FunctionSampler(func=isof)],  #, FunctionSampler(func=lof)
    #     'sample': [RandomUnderSampler()],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
    #     'selection__mode': ['percentile'], #'fpr', 'fdr',
    #     'selection__param': [87, 90, 92],
    #     'selection__score_func': [f_classif, mutual_info_classif],
    #     'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
    #     'estimation__kernel': ['rbf'],
    #     'estimation__C': np.linspace(0.5, 2, num=5),
    #     'estimation__gamma':  ['auto']
    # }, {
    #     'outlier': [FunctionSampler(func=isof)],  #, FunctionSampler(func=lof)
    #     'sample': [RandomUnderSampler()],  #RandomOverSampler(), , SMOTEENN(), SMOTETomek()
    #     'selection__mode': ['percentile'], #'fpr', 'fdr',
    #     'selection__param': [87, 90, 92],
    #     'selection__score_func': [f_classif, mutual_info_classif],
    #     'estimation': [NuSVC(class_weight='balanced', decision_function_shape='ovo')],
    #     'estimation__kernel': ['rbf'],
    #     'estimation__coef0': [48, 50, 52],
    #     'estimation__nu': [0.38, 0.4, 0.42],
    #     'estimation__gamma': ['auto']
    # }
    # ,
    # {
    #     'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
    #     'sample': [RandomUnderSampler()],  # , SMOTE(), ADASYN()
    #     'selection__mode': ['fwe'],
    #     'selection__param': [0.1],
    #     'estimation': [KNeighborsClassifier()],
    #     'estimation__weights': ['uniform', 'distance'],
    #     'estimation__leaf_size': [10, 30, 40],
    #     'estimation__n_neighbors': [3, 5, 10],
    #     'estimation__p': [1, 2, 3],
    #     'estimation__algorithm': ['ball_tree', 'kd_tree']
    #
    # },
    #{
    #     'outlier': [FunctionSampler(func=isof)],  # , FunctionSampler(func=lof)
    #     'sample': [RandomOverSampler()],  # , SMOTE(), ADASYN()
    #     'selection__mode': ['fpr', 'fdr', 'fwe'],
    #     'selection__param': [0.00005, 0.0001, 0.001, 1],
    #     'selection__score_func': [f_classif],
    #     'estimation': [RandomForestClassifier()]
    # }

]

# Gridsearch
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='balanced_accuracy', verbose=4)
=======
svc_search = {
        'outlier': Categorical([FunctionSampler(func=isof)]),  # , FunctionSampler(func=lof)
        'sample': Categorical([RandomUnderSampler()]),  # RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        'selection__mode': ['fwe'],
        'selection__param': Real(1e-2, 1e+2, prior='log-uniform'),
        'model': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'model__kernel': Categorical(['rbf']),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma':  Categorical(['auto'])
    }

svc_search2 = {
        'outlier': Categorical([FunctionSampler(func=isof)]),  # , FunctionSampler(func=lof)
        'sample': Categorical([RandomUnderSampler()]),  # RandomOverSampler(), , SMOTEENN(), SMOTETomek()
        'selection__mode': [None],
        'model': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'model__kernel': Categorical(['rbf']),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma':  Categorical(['auto'])
    }

svc_search3 = {
        'outlier': Categorical([FunctionSampler(func=isof)]),
        'sample': Categorical([RandomUnderSampler()]),
        'selection': [RFE(estimator=DecisionTreeClassifier(), step=10)],  # Set step to 1 in fine tuning!!!
        'selection__n_features_to_select': Integer(20, X_train.shape[1], prior='uniform'),
        'model': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'model__kernel': Categorical(['rbf']),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma':  Categorical(['auto'])
    }


# Hyperparameter opt
opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(svc_search, evals)],
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)
>>>>>>> master
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

# Plot search results
# _ = plot_objective(opt.optimizer_results_[0],
#                    dimensions=["degree", "gamma", "kernel"],
#                    n_minimum_search=int(1e8))
# plt.show()
