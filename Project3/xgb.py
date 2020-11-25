import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import DMatrix,XGBClassifier, XGBRFClassifier
from sklearn.utils import parallel_backend

#load preprepared features
X_train = np.loadtxt('feat/features_train_temp_only.csv', delimiter=',') #preprepared features
# X_train = np.loadtxt('features_train_temp_only.csv', delimiter=',') #preprepared features
print("*")
X_test = np.loadtxt('feat/features_test_temp_only.csv', delimiter=',') #preprepared features
print("*")
print(X_test.shape)
y_train = pd.read_csv('raw/y_train.csv').drop(['id'], axis=1).values
print("*")
Id_test = np.asarray(pd.read_csv('raw/X_test.csv')['id'].values)
print("*")
print(Id_test.shape)

#normalize data
scaler = RobustScaler() #aendern zu normalem?
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("***training***")
#Single classifier
# params={'C': [45.0, 55], 'kernel': ['rbf'], 'gamma': ['scale'],
#         'class_weight':['balanced'] , 'decision_function_shape': ['ovo'] } #rbf svc
# params={'C': [1.0, 10.0, 50.0], 'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5], 'gamma': ['scale'],  'tol': [0.001],
#         'class_weight': ['balanced'], 'decision_function_shape': ['ovo'] }
# estimator=SVC()

# params={'n_estimators': [2000], 'random_state': [0], 'bootstrap': ['True']} #rf
# # estimator=RandomForestClassifier()
# estimator=ExtraTreesClassifier()


# params={'n_neighbors': [7]} #KN
# estimator=KNeighborsClassifier()

# regr = GridSearchCV(estimator, scoring='f1_micro', param_grid=params, cv=10, return_train_score=True)
# regr.fit(X_train, y_train.ravel()) #train the model
# print("***predicting***")
# y_pred = regr.predict(X_test)
# print(regr.best_score_)
# print(regr.best_params_)

#voting with multiple classifier
# param_grid = {
#     'voting': ['soft'],
#     'svc__C': [50],
#     'svc__kernel': ['rbf'],
#     # 'svc__class_weight': ['balanced'],
#     'svc__decision_function_shape': ['ovo']
# }
# classifier = GridSearchCV(
#     VotingClassifier([
#         ('svc', SVC(gamma='scale', probability=True, random_state=0)),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=0))
#     ]),
#     param_grid=param_grid,
#     scoring='f1_micro',
#     n_jobs=-1,
#     cv=10,
#     verbose=1
# )
# classifier.fit(X_train, y_train.ravel())
#
# y_train_pred = classifier.predict(X_train)
# y_pred = classifier.predict(X_test)


# bdt_real = AdaBoostClassifier(
#     RandomForestClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1)

#
# print(f'best test score: {classifier.best_score_}')
# print(f'best parameters: {classifier.best_params_}')
# print(f'training score: {f1_score(y_train_pred, y_train, average="micro")}')

model = XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=200, max_depth=8)
# clf=GridSearchCV(model, cv=10, scoring="f1_micro", iid=False, verbose=10,
#                  param_grid={
#                      'objective': ["multi:softprob"],
#                      'num_class': [4],
#                      'n_estimators': [200],
#                      'max_depth': [8]
#                  }
#                  )
# with parallel_backend(backend='loky'):
#     clf.fit(X_train, y_train.ravel())
model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)

# print(f'best test score: {clf.best_score_}')
# print(f'best parameters: {clf.best_params_}')
# print(f'training score: {f1_score(clf, y_train, average="micro")}')


# print(Id_test.shape)
# print(y_pred.shape)

# -----output---------------
np.hstack(Id_test, y_pred)
out = [Id_test, y_pred] #combine ind & y val
np.savetxt('output.csv', np.transpose(out) , delimiter=',' , header= 'id,y'  ) #print output to file
