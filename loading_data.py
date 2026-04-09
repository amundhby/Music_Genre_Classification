import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo'
]

# Loading data files
df_5s = pd.read_csv('GenreClassData_5s.txt', sep='\t', usecols=features + ['Type'])
df_10s = pd.read_csv('GenreClassData_10s.txt', sep='\t', usecols=features + ['Type'])
df_30s = pd.read_csv('GenreClassData_30s.txt', sep='\t', usecols=features + ['Type'])

# Splitting training and testing data
df_5s_train = df_5s[df_5s['Type'] == 'Train'][features]
df_5s_test  = df_5s[df_5s['Type'] == 'Test'][features]

df_10s_train = df_10s[df_10s['Type'] == 'Train'][features]
df_10s_test  = df_10s[df_10s['Type'] == 'Test'][features]

df_30s_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_test  = df_30s[df_30s['Type'] == 'Test'][features]

train_5s = df_5s_train.to_numpy()
test_5s = df_5s_test.to_numpy()