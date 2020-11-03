import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras import optimizers, utils, initializers, callbacks
from keras.models import Sequential
from keras.layers import Activation, LeakyReLU, Dense, Dropout, BatchNormalization
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
import time

np.random.seed(0)


class Config:
    # TODO: Grid search for all these parameters

    VALID_RATIO = 0.2
    EPOCHS = 100
    QUANTILE_RANGE = (25.0, 75.0)
    LEARNING_RATE = 1e-3
    ACTIVATION = LeakyReLU
    # NEURONS = [256, 64, 16]
    # DROPOUT = [0.4, 0.2, 0.1]
    NEURONS = [64]
    INPUT_DROPOUT = 0.5
    DROPOUT = [0.3]
    DROPOUT_DEFAULT = 0.25
    INITIALISER = initializers.he_uniform()
    EARLY_STOPPING = 25

    # TODO: Test different samplers - SMOTE, SMOTEENN, SMOTETomek, under sampling
    # SAMPLER = SMOTEENN()


def print_done():
    print('Done')


# TODO: Fix this - not same as sklearn
def balanced_accuracy(y_true, y_pred_onehot):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """
    y_true_onehot = utils.to_categorical(y_true)
    # y_pred = K.cast(K.argmax(y_pred_onehot, axis=1), dtype='float32')
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    true_positives = K.sum(K.round(K.clip(y_true_onehot * y_pred_onehot, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true_onehot, 0, 1)), axis=0)
    sensitivity = true_positives / (possible_positives + K.epsilon())
    true_negatives = K.sum(K.round(K.clip((1-y_true_onehot) * (1-y_pred_onehot), 0, 1)), axis=0)
    possible_negatives = K.shape(y_true_onehot)[0] - possible_positives
    specificty = true_negatives / (possible_negatives + K.epsilon())
    accuracy = 0.5 * (sensitivity + specificty)
    bal_acc = K.mean(accuracy)
    return bal_acc


def create_network(n_features):
    model = Sequential()

    # init = False
    if len(Config.DROPOUT) != len(Config.NEURONS):
        print("Dropout size doesn't match neuron layers, setting dropout to 0.25")
        Config.DROPOUT = [Config.DROPOUT_DEFAULT for _ in Config.NEURONS]

    model.add(Dropout(Config.INPUT_DROPOUT, input_shape=(n_features,)))
    for neurons, dropout in zip(Config.NEURONS, Config.DROPOUT):
        # if not init:
        #     model.add(Dense(neurons, input_shape=(n_features,),
        #                     kernel_initializer=Config.INITIALISER))
        #     init = True
        # else:
        #     model.add(Dense(neurons, kernel_initializer=Config.INITIALISER,
        #                     use_bias=False))
        model.add(Dense(neurons, kernel_initializer=Config.INITIALISER,
                        use_bias=False))
        model.add(BatchNormalization())
        model.add(Config.ACTIVATION())
        model.add(Dropout(dropout))

    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=Config.LEARNING_RATE),
                  metrics=['accuracy', balanced_accuracy])
    model.summary()
    return model


def create_early_stopping(monitor):
    return callbacks.EarlyStopping(monitor=monitor, verbose=1,
                                   patience=Config.EARLY_STOPPING, mode='max',
                                   restore_best_weights=True)


def calculate_accuracy(model, data, labels):
    train_pred = np.argmax(model.predict(data), axis=1)
    train_score = balanced_accuracy_score(labels, train_pred)
    print('Balanced accuracy score = {}'.format(train_score))


# Load data
print('Loading data ... \t', end='')
X_out = pd.read_csv('raw/X_test.csv', index_col='id').astype(np.float32)
X_train_full = pd.read_csv('raw/X_train.csv', index_col='id').astype(np.float32)
Y_train_full = pd.read_csv('raw/y_train.csv', index_col='id')
print_done()

# Split data into train and valid, perform input scaling
print('Processing data ... \t', end='')
train_x, valid_x, train_y, valid_y = \
    train_test_split(X_train_full, Y_train_full,
                     test_size=Config.VALID_RATIO, stratify=Y_train_full)
scaler = RobustScaler(quantile_range=Config.QUANTILE_RANGE)
train_x_full_scaled = scaler.fit_transform(X_train_full)
train_x_scaled = scaler.transform(train_x)
valid_x_scaled = scaler.transform(valid_x)
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_train_full),
                                                  np.squeeze(Y_train_full.to_numpy()))
print_done()

# Convert to categorical data
# train_y_cat = utils.to_categorical(train_y)
# valid_y_cat = utils.to_categorical(valid_y)

# Create network and save initial weights
print('Creating network ...')
network = create_network(train_x.shape[1])
network.save_weights('init.h5')

# Create generators for the datasets
# print('Creating generators ... \t', end='')
# train_generator = BalancedBatchGenerator(train_x_scaled, train_y,
#                                          sampler=Config.SAMPLER,
#                                          batch_size=Config.BATCH_SIZE)
# valid_generator = BalancedBatchGenerator(valid_x_scaled, valid_y,
#                                          sampler=Config.SAMPLER,
#                                          batch_size=Config.BATCH_SIZE)
# full_generator = BalancedBatchGenerator(train_x_full_scaled, Y_train_full,
#                                         sampler=Config.SAMPLER,
#                                         batch_size=Config.BATCH_SIZE)
# print_done()

# # Train network using generators and validate simultaneously
# print('Training ...')
# network.fit_generator(generator=train_generator, epochs=Config.EPOCHS,
#                       steps_per_epoch=train_x.shape[0] // Config.BATCH_SIZE,
#                       verbose=1, callbacks=[early_stopping],
#                       validation_data=valid_generator, validation_steps=10)
# print_done()
#
# # Reset weights and train with full data
# network.load_weights('init.h5')
# print('Training full ...')
# network.fit_generator(generator=full_generator, epochs=Config.EPOCHS,
#                       steps_per_epoch=train_x.shape[0] // Config.BATCH_SIZE,
#                       verbose=1)
# print_done()

# Train network with full dataset and validate simultaneously
print('Training ...')
network.fit(x=train_x_scaled, y=train_y, class_weight=class_weights,
            batch_size=train_x_scaled.shape[0], epochs=Config.EPOCHS,
            verbose=1, callbacks=[create_early_stopping('val_balanced_accuracy')],
            validation_data=(valid_x_scaled, valid_y))
print("\nTraining - ", end='')
calculate_accuracy(network, train_x_scaled, train_y)
print("Validation - ", end='')
calculate_accuracy(network, valid_x_scaled, valid_y)
time.sleep(5.0)

# # Reset weights and train with full data
# network.load_weights('init.h5')
# print('Training full ...')
# network.fit(x=train_x_full_scaled, y=Y_train_full, class_weight=class_weights,
#             batch_size=train_x_full_scaled.shape[0],
#             epochs=Config.EPOCHS, verbose=1,
#             callbacks=[create_early_stopping('balanced_accuracy')])
# calculate_accuracy(network, train_x_full_scaled, Y_train_full)
#
# # Perform final prediction
# print('Final prediction ... \t', end='')
# x_out_scaled = scaler.transform(X_out)
# preds = network.predict(x_out_scaled)
# Y_pred = pd.DataFrame(np.argmax(preds, axis=1))
# Y_pred.to_csv('prediction_nn.csv', index_label='id', header=['y'], compression=None)
# print('Results saved as prediction.csv')
