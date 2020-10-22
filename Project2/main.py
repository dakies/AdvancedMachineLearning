import logging
from pprint import pprint
from time import time
import pandas as pd
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
from sklearn.svm import SVR, SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


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
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.99)
    del a, b
    print('Debug mode on')

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('remove const cloumns', VarianceThreshold(threshold=0)),
   # ('outlier', 'passthrough'),
   # ('sample', 'passthrough'),
    ('scale', StandardScaler()),
    ('selection', GenericUnivariateSelect()),
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
       # 'outlier': [FunctionSampler(func=isof), FunctionSampler(func=lof)],  #
       # 'sample': [RandomOverSampler(), RandomUnderSampler(), SMOTEENN(), SMOTETomek()],  #
        'selection__mode': ['fpr', 'fdr', 'fwe'],
        'selection__param': [ 0.0001, 0.001, 1],
        'estimation': [SVC(class_weight='balanced', decision_function_shape='ovo')],
        'estimation__kernel': ['rbf'],
        'estimation__C': [0.001, 0.01, 0.1, 10]
    }#,
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

# Gridsearch
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='balanced_accuracy', cv=2)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe.steps])
print("parameters:")
pprint(param_grid)
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
y_test.to_csv('prediction.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
