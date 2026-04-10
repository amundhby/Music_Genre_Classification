import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kNN_classifier import kNNClassifier
from maxMinScaler import MaxMinScaler
from performance import save_confusion_matrix, plot_histogram

labels = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo',
    'GenreID'
]

# Loading data files
df_30s = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type', 'Genre'])

# Splitting training and testing data
df_30s_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_test  = df_30s[df_30s['Type'] == 'Test'][features]

train_30s = df_30s_train.to_numpy()
test_30s = df_30s_test.to_numpy()

# Task 1
# Scale features to [0, 1] range using Min-Max scaling
scaler = MaxMinScaler(feature_range=(0, 1))

train_30s_scaled = scaler.fit_transform(train_30s[:, :-1])
test_30s_scaled = scaler.transform(test_30s[:, :-1])

classifier = kNNClassifier(k=5)
classifier.fit(train_30s_scaled[:, :-1], train_30s[:, -1])
predictions = classifier.predict(test_30s_scaled[:, :-1])

accuracy = np.mean(predictions == test_30s[:, -1])
print(f'Error rate: {1 - accuracy:.2f}')

# save_confusion_matrix(test_30s_scaled[:, -1], predictions, labels, f"Task 1 - Confusion matrix\n Error rate: {1 - accuracy:.2f}", "task_1_cm")

# Task 2
test_labels = ["pop", "disco", "metal", "classical"]
#plot_histogram(df_30s, test_labels, 'spectral_rolloff_mean', 0, 9000, 16, bins=30, title="Distribution of Spectral Rolloff Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'mfcc_1_mean', -800, 100, 19, bins=30, title="Distribution of MFCC 1 Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'spectral_centroid_mean', 0, 5000, 16, bins=30, title="Distribution of Spectral Centroid Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'tempo', 50, 200, 30, bins=30, title="Distribution of Tempo", task="task_2")

# Task 3
new_feature = 'spectral_rolloff_var'
# spectral_bandwidth_mean
# spectral_contrast_mean
# chroma_stft_1_mean
# spectral_flatness_mean best to now
# 'spectral_rolloff_var' best without scaling

features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    #new_feature,
    'GenreID'
]

df_30s = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type', 'Genre'])
df_30s_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_test  = df_30s[df_30s['Type'] == 'Test'][features]
train_30s = df_30s_train.to_numpy()
test_30s = df_30s_test.to_numpy()

scaler = MaxMinScaler(feature_range=(0, 1))

train_30s_scaled = scaler.fit_transform(train_30s[:, :-1])
test_30s_scaled = scaler.transform(test_30s[:, :-1])

#print(df_30s[df_30s['Genre'] == 'pop'][new_feature])
#plot_histogram(df_30s, test_labels, new_feature, -0.01, 0.15, 60, bins=30, title="Distribution of spectral flatness mean", task="task_3")

classifier = kNNClassifier(k=5)
classifier.fit(train_30s_scaled[:, :-1], train_30s[:, -1])
predictions = classifier.predict(test_30s_scaled[:, :-1])

accuracy = np.mean(predictions == test_30s[:, -1])
print(f'Error rate with new feature: {1 - accuracy:.2f}')

# Task 4
features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo',
    'GenreID'
]

df_5s = pd.read_csv('data/GenreClassData_5s.txt', sep='\t', usecols=features + ['Type', 'Genre'])
df_10s = pd.read_csv('data/GenreClassData_10s.txt', sep='\t', usecols=features + ['Type', 'Genre'])

df_5s_train = df_5s[df_5s['Type'] == 'Train'][features]
df_5s_test  = df_5s[df_5s['Type'] == 'Test'][features]

df_10s_train = df_10s[df_10s['Type'] == 'Train'][features]
df_10s_test  = df_10s[df_10s['Type'] == 'Test'][features]

train_5s = df_5s_train.to_numpy()
test_5s = df_5s_test.to_numpy()

train_10s = df_10s_train.to_numpy()
test_10s = df_10s_test.to_numpy()