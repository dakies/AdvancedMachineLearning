from sklearn import linear_model
from sklearn.metrics import r2_score
import sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
import tensorflow as tf
from keras import losses



#training data
train_x = pd.read_csv("X_train.csv")
train_x.pop('id')
train_y = pd.read_csv("y_train.csv")
train_y.pop('id')

#fill missing data from collom
train_x.fillna(train_x.median(), inplace=True)


# print(train_x.mean(axis=0))

#testing data
test_x = pd.read_csv("X_test.csv")
id = test_x.pop('id')

#fill missing data from TRAINDATA
test_x.fillna(train_x.median(), inplace=True)



# print(train_x)
# print(train_y)
# print(test_x)

# compare feature means of train and test ->teils etwas unterschiedlich
# means_x_train=train_x.mean()
# means_x_test=test_x.mean()
# print("feature means train, test:")
# for (i,j) in zip(means_x_train.to_numpy(),means_x_test.to_numpy()):
#     print(i,  end=" ", flush=True)
#     print(j)

# exclude features with to low variance
#std:832, th=0.02->762, th=0.05-> 720, th=0.1->665, th=0.2->661, th=0.5->660
selector = VarianceThreshold(threshold=0.05)
train_x_lowfeature=selector.fit_transform(train_x)
# print(train_x_lowfeature.shape)
test_x_lowfeature=selector.transform(test_x) #auch x_test anpassen fuer selbes model!
# print(test_x_lowfeature.shape)

# only user k best feature
# local score: 500->0.3
# public score: 500->0.414
kbest = SelectKBest(f_regression, k=150)
# print(train_y.to_numpy().flatten().shape)
train_x_kbest = kbest.fit_transform(train_x_lowfeature, train_y.to_numpy().flatten())
# print(train_x_kbest.shape)
test_x_kbest = kbest.transform(test_x_lowfeature)
# print(test_x_kbest.shape)


# get information about y
# mean=train_y.mean().to_numpy()[0]
# std=train_y.std().to_numpy()[0]
# print(mean)
# print(std)

# #scaling for data with outliers
# norm_train_x=sklearn.preprocessing.robust_scale(train_x_lowfeature)
# norm_train_y=sklearn.preprocessing.robust_scale(train_y).flatten() #+ndarray to array
# norm_test_x=sklearn.preprocessing.scale(test_x_lowfeature)


# print (norm_train_x)
# print (norm_train_y)

# #not normalized data
# std_train_x = train_x.to_numpy()
# std_train_y = train_y.to_numpy().flatten()

# print (std_train_x)
# print (std_train_y)

# Huber Regression
# hub = linear_model.HuberRegressor(epsilon=1.35, max_iter=100, alpha=100, warm_start=False, fit_intercept=False, tol=1e-05)
# hub.fit(train_x_lowfeature, train_y.to_numpy())
#
# w = hub.coef_
# print(hub.score(train_x_lowfeature, train_y))
# print(hub.coef_)
#
# y_pred=hub.predict(test_x_lowfeature)

# print(y_pred)



# #PCA
# ipca = IncrementalPCA(n_components=750)
# ipca.fit(train_x)
# x_train = ipca.transform(train_x)
# x_test = ipca.transform(test_x)
# train_x = pd.DataFrame(data = x_train)
# test_x = pd.DataFrame(data = x_test)

#outlier detection
clf1 = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, behaviour='new')#possible contamination .1, .2, .22; .1 seems to be best
clf1.fit(train_x_kbest)
outliers1=clf1.predict(train_x_kbest)
# print(outliers1.shape)
# print(outliers1)
#for i in range(1212):#1212 is number of data points
    #print(outliers1[i])

#another outlier detection
#cov = EllipticEnvelope(support_fraction=1., contamination='auto')
#arr=np.array(train_x_kbest)
#cov.fit(arr) #does not work since list or array cannot be multiplied with float
#outliers3=cov.predict(arr)


#another outlier detection, which we do not use
clf2 = OneClassSVM(kernel='rbf', gamma=0.05, nu=0.22)
clf2.fit(train_x_kbest)
outliers2=clf2.predict(train_x_kbest)
# print(outliers2.shape)
# print(outliers2)
#for i in range(1212):#1212 is number of data points
    #print(outliers2[i])


train_x_ol=[]
train_y_ol=[]
train_y=train_y.to_numpy()
for i in range(outliers1.shape[0]):
    if outliers1[i]==1:
        train_x_ol.append(train_x_kbest[i])
        train_y_ol.append(train_y[i])

# print(train_x_kbest.shape)
# print(train_y.shape)
# print(np.asarray(train_x_ol).shape)
# print(np.asarray(train_y_ol).shape)
train_x_ol=np.asarray(train_x_ol)
train_y_ol=np.asarray(train_y_ol)

#Scale features using statistics that are robust to outliers
# transformer=RobustScaler()
# transformer.fit(train_x_ol)
# transformer.transform(train_x_ol)
# transformer.fit(train_y_ol)
# transformer.transform(train_y_ol)


# #linear regression
# regr = linear_model.LinearRegression()
# regr.fit(train_x_ol, train_y_ol) #train the model
# y_pred = regr.predict(test_x_kbest)
# # print(y_pred)

# #LASSO CURRENT BEST!
# regr = linear_model.Lasso(alpha=1.0, max_iter=50000)
# regr.fit(train_x_ol, train_y_ol) #train the model
# y_pred = regr.predict(test_x_kbest)

#---------Create Model--------------
model = keras.models.Sequential([

    # keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    # keras.layers.Dropout(rate=0.3),

    # keras.layers.BatchNormalization(),
    # keras.layers.Dense(720, activation=tf.nn.relu),
    # keras.layers.Dropout(rate=0.2),
    #
    # keras.layers.BatchNormalization(),
    # keras.layers.Dense(720, activation=tf.nn.relu),
    # keras.layers.Dropout(rate=0.3),

    keras.layers.Dense(1),
])

# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
model.compile(optimizer='adamax', loss=losses.mean_squared_error, metrics=['accuracy']) #make model ready for training
model.fit(train_x_ol, train_x_ol, epochs=32, shuffle=True, validation_split=0.1,
          verbose=1)
# model.fit(train_x_ol, train_x_ol, epochs=2000, shuffle=True, validation_split=0.1,
#           verbose=1, callbacks=[early_stopping])
print(model.summary())
print(model.predict(test_x_kbest))




# # RANSAC-Regresson
# ransac = linear_model.RANSACRegressor()
# ransac.fit(train_x_ol, train_y_ol)
# y_pred = ransac.predict(test_x_kbest)
#
# print("Score: "+str(r2_score(train_y_ol, ransac.predict(train_x_ol))))


#creating the output file
names = ["id", "y"]
id = pd.DataFrame(id)
result = pd.DataFrame(y_pred)
# result = result*std+mean #REVERSE NORMALISATION

# print(result)

sol = id.join(result)
sol.columns = names

sol.to_csv("submit.csv", index = False)
