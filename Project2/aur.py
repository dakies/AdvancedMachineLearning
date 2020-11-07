# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score






# Assuming number of points is divisible by k

def main():
    #-----Initialise-------------------
    x_train = np.loadtxt(open("raw/X_train.csv", "rb"), delimiter=",", skiprows=1)[:,1:] #import data
    y_train = np.loadtxt(open("raw/Y_train.csv", "rb"), delimiter=",", skiprows=1)[:,1] #import data
    x_test = np.loadtxt(open("raw/X_test.csv", "rb"), delimiter=",", skiprows=1)[:,1:] #import data
    Id_test = np.loadtxt(open("raw/X_test.csv", "rb"), delimiter=",", skiprows=1)[:,0]
    # print(x_train)
    # print(y_train)
    # print(x_test)
    print(Id_test)


    #-----check for imbalance----------
    counter = np.array([0,0,0])
    for i in y_train:
        # print(i)
        counter[int(i)] +=1
    print("%s out of %d" % (counter , np.sum(counter)))
    # output: [ 600 3600  600] out of 4800 ->imbalance (0,1,2)

    nr_points = counter[0]+counter[1]+counter[2]
    class_weights = {0: nr_points/counter[0],
                     1: nr_points/counter[1],
                     2: nr_points/counter[2] }

    # # ---------Scaling----------
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    #---------Train Model--------------
    model = keras.models.Sequential([

        keras.layers.BatchNormalization(),
        keras.layers.Dense(2800, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.3),

        keras.layers.BatchNormalization(),
        keras.layers.Dense(700, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.2),

        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(384, activation=tf.nn.relu),
        # keras.layers.Dropout(rate=0.3),
        #
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(384, activation=tf.nn.relu),
        # keras.layers.Dropout(rate=0.2),

        keras.layers.BatchNormalization(),
        keras.layers.Dense(175, activation=tf.nn.relu),

        keras.layers.Dense(nr_points, activation=tf.nn.softmax),
    ])
    #loss ändern? oder besser so
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #make model ready for training
    model.fit(x_train, y_train, epochs=2000, shuffle=True, validation_split=0.1, class_weight=class_weights,
              verbose=1, callbacks=[early_stopping])
    print("SUMMARY:")
    print(model.summary())

    print("Score on Train: ")
    print(balanced_accuracy_score(y_train, np.argmax(model.predict(x_train), axis=1)))

    y_pred = np.argmax(model.predict(x_test) , axis=1)


    # -----output---------------
    out=[Id_test,y_pred] #combine ind & y val
    np.savetxt('output_tf.csv', np.transpose(out) , delimiter=',' , header= 'id,y' , fmt='%d, %d', comments='') #print output to file


if __name__ =='__main__':
    main()