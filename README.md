# Advanced Machine Learning
## Top 3 methods for task 1
### 1
I outline the steps of our method below:
1. We use KNN imputation to infer the missing values
2. We use SelectKBest to select the top 200 features.
3. We delete the imputed values, and compute them again (with KNN imputer) based on the 200 best features.
4. We standardize the data, apply PCA to the resulting data and visualize the the two principal dimensions. This reveals a cluster surrounded by an outer ring. 
5. We use elliptic outlier detection to filter the data points in the outer ring.
6. Now that we computed the outliers, we apply the steps 1,2,3 again, this time without the outliers.
7. We train a StackedRegressor on the data which consists of the 200 best features of inliers. The StackedRegressor contains support vector regressors with rbf, cosine similarity and laplacian kernels, a DecisionTreeRegressor, an ExtraTreesRegressor, a RandomForestRegressor, a GradientBoostingRegressor and finally a ridge regressor.
8. We tune some hyperparameters of the StackedRegressor using cross validation. The model I am handing in also attained the second best validation score (obtained via 5-fold CV). 

#### 2
The main steps of the approach are as follows:

Feature selection:
- the main criterion for selecting the features was the Pearson correlation with the labels column. The correlations (absolute values) were sorted descendingly and the top ~ 200 most relevant features were selected.
Imputation:
- was done using k-nearest neighbor imputation (KNN), with the number of neighbors considered treated as a hyper parameter.

Outlier detection:
- was done using an ensemble of two algorithms: Isolation Forest and Local Outlier Factor (LOF). The outlier set was determined via the union of the two predicted sets by the algorithms (the intersection version was also tried, but the union did better on cross-validation).
- using an elliptic envelope and a 1-class SVM was also tried, but no improvements were found.
Data normalization:
- features were normalized to have 0 mean and unit variance.
- labels were also normalized (and then model outputs unnormalized)
Model:
- The best results were yielded by tree based gradient boosting. The most important hyperparameters (number of leaves, learning rate, feature fraction) were determined via cross validation. The train-val split percentage was 80%. The LightGBM library was used for implementation.
- Fully Connected Neural Networks were also tried, but they did significantly worse. These were implemented using PyTorch.
### 3
Training and test set are concatenated to enforce the same statistical properties when preprocessing. We impute NaN values with the feature median which seems sensible for a dataset with outliers. We determine the Pearson correlation coefficients between labels and every feature vector for a ranked list of feature importance. Visual inspection of the coefficients suggests to include only 215 features. Inspection of label-feature pairs reveals, that features and labels seem to be correlated linearly and logarithmically. As a feature engineering measure, we add the log-transform of the features. To mitigate the effects of outliers, we remove 20 of them manually by inspection and then take the normalized features (sklearn.preprocessing.StandardScaler) and set every value with an absolute variance larger than 4.6 to zero, i.e., set outliers to the feature mean. We determined this value with cross-validation. Preprocessing is finished by normalizing the features of the dataset with sklearn's standard scaler.
We find via extensive cross-validation testing (sklearn.model.selection) that a Gaussian Process Regressor (sklearn.gaussian_process) with an additive combination of a Matern, an RBF and a white kernel performs best on the data at hand. We use 5-fold CV to find the best parameter nu of the Matern kernel with respect to the R2 score (sklearn.metrics.r2_score). Finally, we fit a model with optimal nu=2.5 and the full training set and output predictions on the test set.
## Top 3 methods for task 2
### 1
I first started by analyzing and visualizing the data. In contrast to the last exercise, the scale and variance of the data is closer this time and after some visualization even seems to be gaussian like distributed. Nevertheless, I use the StandardScaler to scale the data as I use PCA for dimensionality reduction and tranformation of the data. After this step I tried different classifiers such as Boosting, LDA, RandomForests, GMMs, Nearest Neighbors and more. For the class imbalance different strategies where tried such as upsampling, downsapling, SMOTE and other variations of SMOTE. LDA was especially interesting, as the training data could be seperated very nicely. I spend a lot of time trying dimensionality reduction with LDA and then classifying with various strategies which performed very good on the test data. Unfortunately, the projection of the test data performd rather poor as could also be seen in the visualization. In the end A SVM classifier with an rbf kernel and appropriate class weights seems to give the best performance. The hyperparameters are tuned via gridsearch and crossvalidated.
### 2
In terms of preprocessing, first we standardize the data. Next, we apply kernelized principal component analysis with the default RBF kernel - this has outperformed other kernels and all other feature selection techniques we have tried. To handle the class imbalance, we have experimented with various methods in imblearn package: several over, under and mixed sampling approaches have been tested. In the end, undersampling was superior to oversampling and we settled on the Neighborhood Cleaning Rule, as it was yielding the best cross-validation scores. For our model we have explored voting classifiers, but as we've been testing individual models, we've found that an SVM classifier is sufficient to achieve strong performance. Our SVM is using the default RBF kernel and "balanced" class weights, since some of the undersampling techniques do not return a balanced dataset. For evalutation, we've been generating 5 stratified folds, applying the preprocessing pipeline on each pair of training-validation sets and training our model.

### 3
Feature Selection: Kernel PCA('rbf')  from SciKit learn to reduce dimensions down to 400.
Class Imbalance:  We synthesized new samples of the minority class instances in order to equalize the class sizes. The SMOTE algorithm was used for this purpose. Afterwards we downsampled again with randomundersample.
Classifier: We used a stacking classifier:  The meta classifier was a SVC with rbf kernel. The ensemble consisted of three SVC's with a sigmoid a rbf and a polynom kernel respectively.
We ran all our tests trough a 5-fold crossvalidation in order to determine ideal hyperparameters. 
The slow model-generation nature of svc's didn't allow for a fine meshed exploration of the hyperparameters which is why only 7 regularization values per model where explored. 
We could easily see improvements by experimenting with more classifiers inside the ensemble and searching for better hyperparameters. However data with class imbalance is extremely prone to overfitting (probably already the case) and it was decided to not needlessly overoptimize the model. 

## Task 3: Best solutions
### 1
We used an ensemble consisting of the following 3 classifiers:
1. 1D-CNN: We trim/zerp-pad all signals to 17000 points. We then use a CNN with the following architecture. 6 1D-Blocks with decreasing kernel size (1D-Block(32, 6, 3, 0.05), ..., 1D-Block(32, 2, 4, 0.05)) followed by a GlobalMaxPool1D() Dropout(0.1), two Dense(64) and Dense(4, softmax) Layer. (1D-Block(32, 6, 3, 0.05) = [Conv1D(32,6), Conv1D(32,6), MaxPool1D(3), Dropout(0.05)])

2. CNN-LSTM: We extract the rpeaks/heartbeats via biosppy. We then sample 20 segments/beats of length 300 centered around the rpeaks from each signal (zero-padding for shorter samples). Network Input Shape: (20, 300, 1). We use a TimeDistributed CNN with a similar architecture as in (1) but with only 5 1D-Blocks and smaller pool sizes. By using a TimeDistributed wrapper on the CNN we extract features for each heartbeat individually. This is followed by an LSTM(64), Dense(64) and Dense(4, softmax) Layer. 

3. Biosppy Features XGBoost: We use the biosppy package to extract RR duration, QRS Duration, Q amplitude, S amplitude, R amplitude & heartrate for all beats of the signal. We then use mean, std, var, min & max of these features as our input features for our Network. We use a XGBClassifier with 100 estimators.

We use 5 fold bagging and random Oversampling for all 3 classifiers to deal with the class imbalance. Soft voting was used to combine the predictions of the 3 classifiers. 

### 2
In this task, we are asked to handel the ECGs signal with different length. So, our first idea is to extract the essential features from each signal. Here we use the neurokit2 and biosppy. Those two packages could help us to obtain, the amplitude of R, S, Q, P and T, duration of  PR, ST, QT, corrected QT and  Q interval, QRS interval, T interval , RR interval.Those features represent in time series format. And for each one above, the mean, min, max, standard deviation will be caculated. And with help of hrvanalysis and pywt, we extract other features in time and frequancy domain such as heartrate and wavelet decomposition. Finally, we use selected 200 features from those 527 features [with SelectKBest method from Scikit-Learn] and train xgboost classifier.
After that we use a ensemble result from predicted probablity from xgboost classifier and a neural network. the network refers to "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network"(Awni Y. Hannun). The public score got better than single method;
### 3
The non-equal length ECG signals are zero-padded and denoised by applying a band-pass filter. We use signals.ecg from the biosppy package for this task. As scaling the data by features destroys temporal information in the timeseries, we use rpeaks from signals.ecg to identify QRS complexes and instead scale every signal with the median of its R peaks (robust against outliers).
Next, we attempted manual feature extraction with hand-crafted filters but soon realized that a DNN with 1D conv layers fits the task perfectly. After a literature search, we found inspirational ideas in the paper www.researchgate.net/publication/326579638. We build a DNN with blocks of conv1D layers with ReLU (for feature extraction), batch normalization (for stabilization) and dropout (to prevent overfitting). Additionally, we use max pooling halving the feature space in every block and decrease filter and channel sizes in deeper layers to speed up training, arriving at a 13-block DNN. We use Keras with the Adam optimizer, a random test split of 20% for training, batchsize of 10 and apply data augmentation by randomly shifting timeseries up to 20 time units to the left. We train for 150 epoch, with early stopping on the test micro averaged F1 score and inverse proportional class weights to adjust for class imbalance. Finally, we exploit the randomness in the model (random initialization and splits), train 13 different DNNs and combine the predictions in a majority vote ensemble with random tie break
## Task 4: Top 3 solutions
### 1
At first, we decided to train a deep learning model consisting of CNN and LSTM layers. CNN layers were supposed to perform feature extraction and LSTM layers to capture stage transitions. Each input was fed to two different sets of CNNs with 4 layers. The specification of each set is given below:
1st CNN ==  out-channels = [64, 128, 128, 128] -- kernel_size = [64, 8, 8, 8] -- strides = [32, 1, 1, 1] 
2nd CNN ==  out-channels = [256, 128, 128, 128] -- kernel_size = [8, 8, 8, 8] -- strides = [4, 1, 1, 1] 
The idea is that a small filter can capture time-domain features, and a large filter can capture frequency-domain features.
The outputs of CNNs were concatenated and given to two layers of bidirectional LSTM cells with 512 hidden units. After training this model, we discovered that CNNs could predict the output very well, and LSTM cells didn't add that much expressive power to the model. Hence, we decided to only use CNNs in the final model. To better estimate the temporal dependencies, at each time t, we concatenated the signals at time t-2, t-1, t, t+1, and t+2 and gave this sequence to the CNNs. An initial learning rate of 5e-4 was used for final training and was reduced by a factor of 0.5 if the validation loss was not improving. We also used weight decay = 1e-3 to prevent overfitting. Finally, to compensate for class imbalances, for each training mini-batch, we computed the class weights and performed cost-sensitive updates with cross-entropy loss.

### 2
For each epoch, we computed a spectrogram for the three different channels. Based on the spectrogram we computed 57 different features based on the different frequency classes. Examples are alpha/beta/gamma/delta/theta/total  {power, magnitude, power ratio wrt total power} or number of zero crossings in the spectrogram. In addition, we applied log-standardization on a per-mice basis to reduce differences between the five mices.  
To exploit temporal dependencies of the epochs, we attach the features of the neighboring epochs to each epoch. This allows us to treat each input sample to a model as a sequence of epochs.
Our final submission consists of a vote using 27 submissions that use a variety of classifiers such as SVM, Gradient Boosting Trees, multilayer perceptrons and Recurrent Neural Networks. The parameter for each of these classifiers were determined using leave-one-subject-out CV. Out of all these classifiers, recurrent neural networks using LSTM cells performed the best, given their "native" ability to handle sequenced data. We used a variety of setups for our recurrent networks since tweaking the hyper-parameters (number of layers, units per layer, regularization, etc.) was difficult due to the low number of training mices. 
Before conducting the vote, we corrected each submission for stage transitions that are infeasible such as direct wake-->rem transitions based on simple replacement rules.  
After the vote, we again applied such a correction.

### 3
For the feature extraction we mainly used the biosppy.eeg.eeg and biosppy.emg.find_onsets methods. The eeg method gave us the power of several frequency bands, for each of which we added some characterizing statistics to the features. Additionally, we used welch’s method and a quadrature rule to compute the relative power of each frequency band (and computed the frequencies where 50% and 95% of the power of the signal are reached. For the emg signal we first filtered it and then computed features like the average distance of onsets, min/max of peaks, no of onsets and others.
In order to use the time dependency we also included features from observations before/after each ID to the features of each ID. By trying out different combinations we found the best combination was to use the features of the ID and the ones of the four preceeding IDs.
We scaled the data with sklearn.StandardScaler subject-by-subject to account for inter-subject variance.
For the classification we used LightGBMs LGBMClassifier and used sklearn.GridSearchCV to find the best parameters. After a first round of LGB classification we added the the prediction of the ID and a couple of observations before/after each ID to the feature set and trained another LGBMClassifier, this improve
