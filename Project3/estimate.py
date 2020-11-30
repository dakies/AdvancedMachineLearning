from pprint import pprint
from time import time

import matplotlib.pyplot as plt
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
from xgboost import XGBClassifier, plot_importance


def lof(x, y):
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def isof(x, y, cont='auto'):
    # Range: [0, 0.5]
    model = IsolationForest(contamination=cont)
    y_pred = model.fit_predict(x)
    return x[y_pred == 1], y[y_pred == 1]


def rmcorr(x_train, x_test):
    # Remove correlated features
    # Has a bug
    correlated_features = set()
    correlation_matrix = x_train.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    print('Correlated features')
    print(len(correlated_features))

    X_train = x_train.drop(labels=correlated_features, axis=1, inplace=True)
    X_test = x_test.drop(labels=correlated_features, axis=1, inplace=True)
    return X_train, X_test


# Load data
X_test = pd.read_csv('final/test_127.csv')
X_train = pd.read_csv('final/train_127.csv')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

caner_test = pd.read_csv('feat_caner/features_test_43.csv')
caner_train = pd.read_csv('feat_caner/features_train_43.csv')

# Todo how many infs and nans are there?
# si = SimpleImputer(strategy='median')
# X_train = si.fit_transform(X_train)
# X_test = si.transform(X_test)

temp = pd.DataFrame()
temp2 = pd.DataFrame()
percent_missing = X_train.isnull().sum() * 100 / len(X_train)
missing_value_df = pd.DataFrame({'column_name': X_train.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)

# print(missing_value_df)

# for label, content in X_train.iteritems():
#    if not X_train[label].isnull().values.any():
#        temp[label] = X_train[label]
#        temp2[label] = X_test[label]
#    else:
#        print(label)

X_train = X_train.drop(
    ['HRV_LFn', 'HRV_SampEn', 'HRV_LF', 'HRV_LFHF', 'HRV_VLF', 'HRV_ULF', 'heart_rate', 'HRV_HF', 'HRV_VHF', 'HRV_HFn',
     'HRV_LnHF'], axis=1)
X_test = X_test.drop(
    ['HRV_LFn', 'HRV_SampEn', 'HRV_LF', 'HRV_LFHF', 'HRV_VLF', 'HRV_ULF', 'heart_rate', 'HRV_HF', 'HRV_VHF', 'HRV_HFn',
     'HRV_LnHF'], axis=1)

X_train = pd.concat([X_train, caner_train.iloc[:, 28:]], axis=1)
X_test = pd.concat([X_test, caner_test.iloc[:, 28:]], axis=1)
# X_train, X_test = rmcorr(X_train, X_test)

print(X_train.isnull().values.any())
print(y_train.isnull().values.any())
print()
print(X_train.shape)
print(y_train.shape)

pseudotest = False
if pseudotest:
    X_train, X_test_val, y_train, y_test_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Reduce Data for debugging
evals = 10
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
    'model': [XGBClassifier()],
    'scale': ['passthrough'],
    # 'model__min_child_weight': (0, 10),
    # 'model__max_delta_step': Integer(0, 20),
    # 'model__colsample_bytree': (0.01, 1.0, 'uniform'),
    # 'model__colsample_bylevel': (0.01, 1.0, 'uniform'),
    'model__learning_rate': (0.01, 1.0, 'log-uniform'),
    'model__n_estimators': Integer(50, 150),
    'model__scale_pos_weight': Real(1, 1000, 'log-uniform'),
    'model__min_child_weight': Integer(1, 10),
    'model__gamma': Integer(1, 5),
    'model__subsample': Real(0.3, 1),
    'model__colsample_bytree': Real(0.3, 1),
    'model__max_depth': Integer(3, 20)
}

xg = XGBClassifier()
xg.load_model('model_file_name.json')
xgb_search_prev = {
    'model': [xg],
    # 'model__learning_rate': (0.01, 1.0, 'log-uniform'),
    # 'model__min_child_weight': (0, 10),
    # 'model__max_delta_step': Integer(0, 20),
    # 'model__colsample_bytree': (0.01, 1.0, 'uniform'),
    # 'model__colsample_bylevel': (0.01, 1.0, 'uniform'),

    # 'model__n_estimators': Integer(100, 200),
    # 'model__scale_pos_weight': Real(1, 1000, 'log-uniform'),
    # 'model__min_child_weight': Integer(1, 10),
    # 'model__gamma': Integer(1, 5),
    # 'model__subsample': Real(0.3, 1),
    # 'model__colsample_bytree': Real(0.1, 1),
    # 'model__max_depth': Integer(6, 12)
}

xgb_pca_search = {
    # , FunctionSampler(func=lof)
    'selection': [PCA(n_components='mle')],
    'model': [XGBClassifier()],
    'model__scale_pos_weight': Integer(1, 100)
}

# Hyper-parameter opt
opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(xgb_search_prev, evals)],
    cv=3,
    scoring='f1_micro',
    n_jobs=-1,
    verbose=3
)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe.steps])
print("parameters:")
pprint(xgb_search)
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

# Save tree
# https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
opt.best_estimator_['model'].save_model('model_file_name.json')

plot_importance(opt.best_estimator_['model'], ax=plt.gca(), max_num_features=15)
plt.show()
