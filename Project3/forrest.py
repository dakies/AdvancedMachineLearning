import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler



#load preprepared features
X_train_1 = np.loadtxt('features_train.csv', delimiter=',') #preprepared features
X_train_2 = np.loadtxt('features_train_temp_only.csv', delimiter=',') #preprepared features
# X_train = np.loadtxt('features_train_temp_only.csv', delimiter=',') #preprepared features
print("*")
X_test_1 = np.loadtxt('features_test.csv', delimiter=',') #preprepared features
X_test_2 = np.loadtxt('features_test_temp_only.csv', delimiter=',') #preprepared features
print("*")
# print(X_test.shape)
y_train = pd.read_csv('mitbih_train.csv.csv').drop(['id'], axis=1).values
print("*")
Id_test = np.asarray(pd.read_csv('mitbih_test.csv')['id'].values)
print("*")
# print(Id_test.shape)

#normalize data
scaler_1 = RobustScaler() #aendern zu normalem?
X_train_1 = scaler_1.fit_transform(X_train_1)
X_test_1 = scaler_1.transform(X_test_1)

scaler_2 = RobustScaler() #aendern zu normalem?
X_train_2 = scaler_2.fit_transform(X_train_2)
X_test_2 = scaler_2.transform(X_test_2)


print("***training***")
#Single classifier
# params={'C': [45.0, 55], 'kernel': ['rbf'], 'gamma': ['scale'],
#         'class_weight':['balanced'] , 'decision_function_shape': ['ovo'] } #rbf svc
# params={'C': [1.0, 10.0, 50.0], 'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5], 'gamma': ['scale'],  'tol': [0.001],
#         'class_weight': ['balanced'], 'decision_function_shape': ['ovo'] }
# estimator=SVC()

# params={'n_estimators': [2000], 'random_state': [0], 'bootstrap': ['True']} #rf
estimator_1 = RandomForestClassifier(n_estimators=1500, random_state=0)
estimator_1.fit(X_train_1, y_train.ravel())
y_prob_1 = estimator_1.predict_proba(X_test_1)
print(y_prob_1[0])
print(y_prob_1[1])

estimator_2 = RandomForestClassifier(n_estimators=1500, random_state=0)
estimator_2.fit(X_train_2, y_train.ravel())
y_prob_2 = estimator_2.predict_proba(X_test_2)
print(y_prob_2[0])
print(y_prob_2[1])

y_prob = y_prob_1+y_prob_2
print(y_prob[0])
print(y_prob[1])

y_pred=[]
for entry in y_prob:
    y_pred.append(np.argmax(entry))
y_pred=np.asarray(y_pred)
print(y_pred[0])
print(y_pred[1])

# print ("**************")
# for i,j in zip(y_prob_1, y_prob_2):
#     print (i)
#     print (j)
#     print(" ")
# print ("**************")


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
#
# print(f'best test score: {classifier.best_score_}')
# print(f'best parameters: {classifier.best_params_}')
# print(f'training score: {f1_score(y_train_pred, y_train, average="micro")}')




# print(Id_test.shape)
# print(y_pred.shape)

# -----output---------------
out=[Id_test,y_pred] #combine ind & y val
np.savetxt('output_geteilt.csv', np.transpose(out) , delimiter=',' , header= 'id,y' , fmt='%d, %d', comments='') #print output to file
