import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kNN_classifier import kNNClassifier
from sklearn.metrics import confusion_matrix

labels = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo',
    'GenreID'
]

# Loading data files
df_5s = pd.read_csv('data/GenreClassData_5s.txt', sep='\t', usecols=features + ['Type'])
df_10s = pd.read_csv('data/GenreClassData_10s.txt', sep='\t', usecols=features + ['Type'])
df_30s = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type'])

# Splitting training and testing data
df_5s_train = df_5s[df_5s['Type'] == 'Train'][features]
df_5s_test  = df_5s[df_5s['Type'] == 'Test'][features]

df_10s_train = df_10s[df_10s['Type'] == 'Train'][features]
df_10s_test  = df_10s[df_10s['Type'] == 'Test'][features]

df_30s_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_test  = df_30s[df_30s['Type'] == 'Test'][features]

train_5s = df_5s_train.to_numpy()
test_5s = df_5s_test.to_numpy()

train_10s = df_10s_train.to_numpy()
test_10s = df_10s_test.to_numpy()

train_30s = df_30s_train.to_numpy()
test_30s = df_30s_test.to_numpy()

classifier = kNNClassifier(k=5)
classifier.fit(train_30s[:, :-1], train_30s[:, -1])
predictions = classifier.predict(test_30s[:, :-1])
accuracy = np.mean(predictions == test_30s[:, -1])
print(f'Error rate: {1 - accuracy:.2f}')

cm = confusion_matrix(
    test_30s[:, -1],
    predictions,
    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

print(cm)