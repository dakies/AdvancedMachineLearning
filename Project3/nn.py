import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

# load preprepared features
X_train = np.loadtxt('features_train_extended.csv', delimiter=',')  # preprepared features
# X_train = np.loadtxt('features_train_temp_only.csv', delimiter=',') #preprepared features
print("*")
X_test = np.loadtxt('features_test_extended.csv', delimiter=',')  # preprepared features
print("*")
# print(X_test.shape)
y_train = pd.read_csv('y_train.csv').drop(['id'], axis=1).values
print("*")
Id_test = np.asarray(pd.read_csv('X_test.csv')['id'].values)
print("*")
# print(Id_test.shape)

# normalize data
scaler = RobustScaler()  # aendern zu normalem?
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)

weights = {0: 5117 / (4 * 3030),
           1: 5117 / (4 * 443),
           2: 5117 / (4 * 1474),
           3: 5117 / (4 * 170)}

# ---------Create Model--------------
model = keras.models.Sequential([
    # keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(120, activation=tf.nn.relu),
    # keras.layers.Conv1D(64, 120, activation='relu')
    # keras.layers.MaxPooling1D()
    # keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

    keras.layers.BatchNormalization(),
    keras.layers.Dense(1024, activation=tf.nn.sigmoid),
    keras.layers.Dropout(rate=0.5),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.sigmoid),
    keras.layers.Dropout(rate=0.5),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dropout(rate=0.5),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dropout(rate=0.5),

    # keras.layers.BatchNormalization(),
    # keras.layers.Dense(256, activation=tf.nn.relu),
    # keras.layers.Dropout(rate=0.3),
    #
    # keras.layers.BatchNormalization(),
    # keras.layers.Dense(256, activation=tf.nn.relu),
    # keras.layers.Dropout(rate=0.2),

    keras.layers.BatchNormalization(),
    # keras.layers.Dense(64, activation=tf.nn.sigmoid),

    keras.layers.Dense(4, activation=tf.nn.softmax),
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='auto')
# LOSS ANDER??
model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # make model ready for training
model.fit(X_train, y_train, epochs=500, shuffle=True, validation_split=0.1, class_weight=weights,
          verbose=1)  # , callbacks=[early_stopping]
print(model.summary())

# -------Use & Evaluate trained model------
y_train_pred = model.predict([X_train])
y_train_pred = np.argmax(y_train_pred, axis=1)
# print("y_train and predicted: ", y_train, y_train_pred)

acc_train = accuracy_score(y_train, y_train_pred)
print("Train Accuracy: ", acc_train)

y_test_pred = model.predict([X_test])
y_test_pred = np.argmax(y_test_pred, axis=1)

# -----output---------------
out = [Id_test, y_test_pred]  # combine ind & y val
np.savetxt('output_nn.csv', np.transpose(out), delimiter=',', header='id,y', fmt='%d, %d',
           comments='')  # print output to file
