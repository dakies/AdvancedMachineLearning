import pandas as pd
import numpy as np

# Load data
train_eeg1 = pd.read_csv('raw/train_eeg1.csv', index_col='id')
train_eeg2 = pd.read_csv('raw/train_eeg2.csv', index_col='id')
train_emg = pd.read_csv('raw/train_emg.csv', index_col='id')
test_eeg1 = pd.read_csv('raw/test_eeg1.csv', index_col='id')
test_eeg2 = pd.read_csv('raw/test_eeg2.csv', index_col='id')
test_emg = pd.read_csv('raw/test_emg.csv', index_col='id')


