import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras import optimizers, utils, initializers, callbacks
from keras.models import Sequential
from keras.layers import Activation, LeakyReLU, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, classification_report
from sklearn.utils import class_weight
import time

np.random.seed(0)


class Config:
    VALID_RATIO = 0.2
    EPOCHS = 500
    QUANTILE_RANGE = (25.0, 75.0)
    EARLY_STOPPING = 25
    CROSS_VAL = 4
    VERBOSE = 0

    # TODO: Test different samplers - SMOTE, SMOTEENN, SMOTETomek, under sampling
    # SAMPLER = SMOTEENN()


Params = {'NEURONS': [(128, 32, 16)],
          'DROPOUT': [.2],
          'ACTIVATION': ['relu'],
          'INPUT_DROPOUT': [0.3],
          'LEARNING_RATE': [1e-3],
          'INITIALISER': ['he_uniform']}
# Params = {'NEURONS': [(128, 32, 16), (128, 32), (128, 16), (64, 16), (64, 8), (64,), (32,)],
#           'DROPOUT': np.linspace(0.1, 0.3, 3),
#           'ACTIVATION': ['relu', 'tanh'],
#           'INPUT_DROPOUT': np.linspace(0.0, 0.3, 4),
#           'LEARNING_RATE': np.logspace(-4., -2., 3),
#           'INITIALISER': ['he_uniform']}


def print_done():
    print('Done')


def balanced_accuracy(y_true_onehot, y_pred_onehot):
    """
    Computes the balanced accuracy metric for a multi-class classification problem
    https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """
    true_positives = K.sum(K.round(K.clip(y_true_onehot * y_pred_onehot, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true_onehot, 0, 1)), axis=0)
    sensitivity = true_positives / (possible_positives + K.epsilon())
    true_negatives = K.sum(K.round(K.clip((1 - y_true_onehot) * (1 - y_pred_onehot), 0, 1)), axis=0)
    possible_negatives = K.sum(K.round(K.clip(1 - y_true_onehot, 0, 1)), axis=0)
    specificty = true_negatives / (possible_negatives + K.epsilon())
    accuracy = 0.5 * (sensitivity + specificty)
    bal_acc = K.sum(accuracy * possible_positives) / K.sum(possible_positives)
    return bal_acc


def create_network(NEURONS, DROPOUT, ACTIVATION, INPUT_DROPOUT,
                   LEARNING_RATE, INITIALISER, N_FEATURES):
    init = initializers.he_uniform() if (INITIALISER == 'he_uniform') else INITIALISER

    model = Sequential()
    model.add(Dropout(INPUT_DROPOUT, input_shape=(N_FEATURES,)))
    for neurons_layer in NEURONS:
        model.add(Dense(neurons_layer, kernel_initializer=init,
                        use_bias=False))
        model.add(BatchNormalization())
        if ACTIVATION == 'leaky_relu':
            model.add(LeakyReLU())
        else:
            model.add(Activation(ACTIVATION))
        model.add(Dropout(DROPOUT))

    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy', balanced_accuracy])
    if Config.VERBOSE: model.summary()
    return model


def create_early_stopping(monitor):
    return callbacks.EarlyStopping(monitor=monitor, verbose=Config.VERBOSE,
                                   patience=Config.EARLY_STOPPING, mode='max',
                                   restore_best_weights=True)


def calculate_accuracy(model, data, labels):
    train_pred = np.argmax(model.predict(data), axis=1)
    train_score = balanced_accuracy_score(labels, train_pred)
    print('Balanced accuracy score = {}'.format(train_score))


def balanced_accuracy_metric(y_true, y_pred):
    return balanced_accuracy_score(y_true=np.argmax(y_true, axis=1), y_pred=y_pred)


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
x_out_scaled = scaler.transform(X_out)
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_train_full),
                                                  np.squeeze(Y_train_full.to_numpy()))
Params['N_FEATURES'] = [train_x.shape[1]]
print_done()

# Convert to categorical data
train_y_cat = utils.to_categorical(train_y)
valid_y_cat = utils.to_categorical(valid_y)
train_y_full_cat = utils.to_categorical(Y_train_full)

classifier = KerasClassifier(build_fn=create_network, verbose=Config.VERBOSE)
scorer = make_scorer(balanced_accuracy_metric, greater_is_better=True)
grid = GridSearchCV(estimator=classifier, param_grid=Params,
                    scoring=scorer, cv=Config.CROSS_VAL)
grid_result = grid.fit(X=train_x_scaled, y=train_y_cat, class_weight=class_weights,
                       batch_size=train_x_scaled.shape[0], epochs=Config.EPOCHS,
                       verbose=Config.VERBOSE, callbacks=[create_early_stopping('val_balanced_accuracy')],
                       validation_data=(valid_x_scaled, valid_y_cat))
best_model = grid_result.best_estimator_

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
time.sleep(5.0)

# Train best estimator on whole data
# print('Training full ...')
# best_model.fit(x=train_x_full_scaled, y=train_y_full_cat,
#                class_weight=class_weights,
#                batch_size=train_x_full_scaled.shape[0],
#                epochs=Config.EPOCHS, verbose=Config.VERBOSE,
#                callbacks=[create_early_stopping('balanced_accuracy')])
train_y_pred = best_model.predict(train_x_scaled)
print('Training report:')
print(classification_report(y_true=train_y, y_pred=train_y_pred))
val_y_pred = best_model.predict(valid_x_scaled)
print('\nValidation report:')
print(classification_report(y_true=valid_y, y_pred=val_y_pred))

# Perform final prediction
print('Final prediction ... \t', end='')
preds = best_model.predict(x_out_scaled)
Y_pred = pd.DataFrame(preds)
Y_pred.to_csv('prediction_nn.csv', index_label='id', header=['y'], compression=None)
print('Results saved as prediction.csv')
