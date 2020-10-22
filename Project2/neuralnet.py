import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from imblearn.keras import BalancedBatchGenerator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import balanced_accuracy_score


def make_model(n_features):
    model = Sequential()
    model.add(Dense(200, input_shape=(n_features,),
              kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def fit_predict_balanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, y_train,
                                                batch_size=1000,
                                                random_state=42)
    model.fit_generator(generator=training_generator, epochs=5, verbose=1)
    y_pred = model.predict_proba(X_test, batch_size=1000)
    return balanced_accuracy_score(y_test, y_pred)


# Load data
sample = pd.read_csv('raw/sample.csv')
X_test = pd.read_csv('raw/X_test.csv', index_col='id')
X_train = pd.read_csv('raw/X_train.csv', index_col='id')
y_train = pd.read_csv('raw/y_train.csv', index_col='id')

pipeline = make_pipeline(
    StandardScaler())
model = make_model(X_train.shape[1])
training_generator = BalancedBatchGenerator(X_train, y_train,
                                            batch_size=1000)
model.fit_generator(generator=training_generator, epochs=500, verbose=1)
y_pred = model.predict_proba(X_test, batch_size=1000)