from pprint import pprint
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier
from xgboost import plot_importance


def lof(x, y):
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def isof(x, y):
    model = IsolationForest()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


# Load data
X_test = pd.read_csv('feat2/test_63.csv')
X_train = pd.read_csv('feat2/train_63.csv')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

# Todo how many infs and nans are there?
X_train = X_train[X_train.columns[np.isfinite(X_train).all(0)]]
X_test = X_test[X_train.columns[np.isfinite(X_train).all(0)]]
print(X_train.isnull().values.any())
print(y_train.isnull().values.any())
# temp = pd.DataFrame()
# temp2 = pd.DataFrame()
# for label, content in X_train.iteritems():
#     if not X_train[label].isnull().values.any():
#         temp[label] = X_train[label]
#         temp2[label] = X_test[label]
# X_train = temp
# X_test = temp2


pseudotest = True
if pseudotest:
    X_train, X_test_val, y_train, y_test_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# Reduce Data for debugging
evals = 40
if 0:
    [X_train, a, y_train, b] = train_test_split(X_train, y_train, test_size=0.9)
    del a, b
    evals = 3
    print('Debug mode on')
# Todo remove highly correlated features
# Todo features selction

# Create pipeline
pipe = Pipeline([
    # the scale stage is populated by the param_grid
    ('outlier', 'passthrough'),
    ('selection', 'passthrough'),
    ('scale', RobustScaler()),
    ('model', 'passthrough')
])
# Specify parameters to be searched over
svc_search = {
    'outlier': Categorical([FunctionSampler(func=isof), FunctionSampler(func=lof)]),  # , FunctionSampler(func=lof)
    'model': [SVC(class_weight='balanced', decision_function_shape='ovo')],
    'model__kernel': Categorical(['rbf']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform')

}

xgb_search = {
    'outlier': Categorical([FunctionSampler(func=isof)]),  # , FunctionSampler(func=lof)
    'model': [XGBClassifier()],
    # 'model__learning_rate': (0.01, 1.0, 'log-uniform'),
    # 'model__min_child_weight': (0, 10),
    # 'model__max_depth': Integer(0, 50),
    # 'model__max_delta_step': Integer(0, 20),
    # 'model__subsample': (0.01, 1.0, 'uniform'),
    # 'model__colsample_bytree': (0.01, 1.0, 'uniform'),
    # 'model__colsample_bylevel': (0.01, 1.0, 'uniform'),
    # 'model__gamma': (1e-9, 0.5, 'log-uniform'),
    # 'model__n_estimators': Integer(50, 100),
    'model__scale_pos_weight': (1, 1000, 'log-uniform')
}

xgb_pca_search = {
    'outlier': Categorical([FunctionSampler(func=isof), FunctionSampler(func=lof)]),  # , FunctionSampler(func=lof)
    'selection': [PCA(n_components='mle')],
    'model': [XGBClassifier()],
    'model__scale_pos_weight': Integer(1, 100)
}

# Hyper-parameter opt
opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(xgb_search, evals), (xgb_pca_search, evals)],
    cv=3,
    scoring='f1_micro',
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

if pseudotest:
    y_pred = opt.best_estimator_.predict(X_test_val)
    F1 = f1_score(y_test_val, y_pred, average='micro')
    print('Score on pseudo test set')
    print(F1)

# Predict for test set
y_test = opt.best_estimator_.predict(X_test)
print()

# Save prediction
y_test = pd.DataFrame(y_test)
y_test.to_csv('prediction.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')

plot_importance(opt.best_estimator_['model'], ax=plt.gca())
plt.show()
