import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kNN_classifier import kNNClassifier
from maxMinScaler import MaxMinScaler
from QDA_classifier import QDAClassifier
from LDA_classifier import LDACLassifier
from performance import save_confusion_matrix, plot_histogram, findBestNewFeature, findBestNewFeatureQDA

labels = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo'
]

# Loading data files
df_30s = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type', 'GenreID', 'Genre'])

# Splitting training and testing data
df_30s_X_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_Y_train = df_30s[df_30s['Type'] == 'Train']['GenreID']
df_30s_X_test  = df_30s[df_30s['Type'] == 'Test'][features]
df_30s_Y_test  = df_30s[df_30s['Type'] == 'Test']['GenreID']

X_train_30s = df_30s_X_train.to_numpy()
Y_train_30s = df_30s_Y_train.to_numpy()
X_test_30s  = df_30s_X_test.to_numpy()
Y_test_30s  = df_30s_Y_test.to_numpy()

# Task 1
scaler = MaxMinScaler(feature_range=(0, 1))

scaled_X_train_30s = scaler.fit_transform(X_train_30s)
scaled_X_test_30s = scaler.transform(X_test_30s)

classifier = kNNClassifier(k=5)
classifier.fit(scaled_X_train_30s, Y_train_30s)
predictions = classifier.predict(scaled_X_test_30s)

accuracy = np.mean(predictions == Y_test_30s)
print(f'Error rate: {1 - accuracy:.2f}')

# UNCOMMENT BELOW IF YOU WANT TO SAVE A NEW CONFUSION MATRIX
#save_confusion_matrix(Y_test_30s, predictions, labels, f"Task 1 - Confusion matrix\n Error rate: {1 - accuracy:.2f}", "task_1_cm")


# Task 2
test_labels = ["pop", "disco", "metal", "classical"]
# UNCOMMENT BELOW IF YOU WANT TO SAVE NEW DISTRIBUTION PLOTS
#plot_histogram(df_30s, test_labels, 'spectral_rolloff_mean', 0, 9000, 16, bins=30, title="Distribution of Spectral Rolloff Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'mfcc_1_mean', -800, 100, 19, bins=30, title="Distribution of MFCC 1 Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'spectral_centroid_mean', 0, 5000, 16, bins=30, title="Distribution of Spectral Centroid Mean", task="task_2")
#plot_histogram(df_30s, test_labels, 'tempo', 50, 200, 30, bins=30, title="Distribution of Tempo", task="task_2")


# Task 3
remaining_features = ["zero_cross_rate_mean", "zero_cross_rate_std", "rmse_mean",
                     "rmse_var", "spectral_centroid_var", "spectral_bandwidth_mean",
                     "spectral_bandwidth_var", "spectral_rolloff_var", "spectral_contrast_mean",
                     "spectral_contrast_var", "spectral_flatness_mean", "spectral_flatness_var",
                     "chroma_stft_1_mean", "chroma_stft_2_mean", "chroma_stft_3_mean",
                     "chroma_stft_4_mean", "chroma_stft_5_mean", "chroma_stft_6_mean",
                     "chroma_stft_7_mean", "chroma_stft_8_mean", "chroma_stft_9_mean",
                     "chroma_stft_10_mean", "chroma_stft_11_mean", "chroma_stft_12_mean",
                     "chroma_stft_1_std", "chroma_stft_2_std", "chroma_stft_3_std",
                     "chroma_stft_4_std", "chroma_stft_5_std", "chroma_stft_6_std",
                     "chroma_stft_7_std", "chroma_stft_8_std", "chroma_stft_9_std",
                     "chroma_stft_10_std", "chroma_stft_11_std", "chroma_stft_12_std",
                     "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean", "mfcc_6_mean",
                     "mfcc_7_mean", "mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean",
                     "mfcc_12_mean", "mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std", 
                     "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", 
                     "mfcc_10_std", "mfcc_11_std", "mfcc_12_std"]

prev_features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
df_30s_full = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=prev_features + ['Type', 'GenreID', 'Genre'] + remaining_features)
features, new_feature, index_of_replaced_feature = findBestNewFeature(remaining_features, prev_features, df_30s_full)

df_30s = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type', 'GenreID', 'Genre'])

df_30s_X_train = df_30s[df_30s['Type'] == 'Train'][features]
df_30s_Y_train = df_30s[df_30s['Type'] == 'Train']['GenreID']
df_30s_X_test  = df_30s[df_30s['Type'] == 'Test'][features]
df_30s_Y_test  = df_30s[df_30s['Type'] == 'Test']['GenreID']

X_train_30s = df_30s_X_train.to_numpy()
Y_train_30s = df_30s_Y_train.to_numpy()
X_test_30s  = df_30s_X_test.to_numpy()
Y_test_30s  = df_30s_Y_test.to_numpy()

scaler = MaxMinScaler(feature_range=(0, 1))

scaled_X_train_30s = scaler.fit_transform(X_train_30s)
scaled_X_test_30s = scaler.transform(X_test_30s)

classifier = kNNClassifier(k=5)
classifier.fit(scaled_X_train_30s, Y_train_30s)
predictions = classifier.predict(scaled_X_test_30s)

accuracy = np.mean(predictions == Y_test_30s)

print(f'Error rate with new feature: {1 - accuracy:.2f}')
print(f'Feature with best accuracy: {new_feature}')
print(f'It replaced: {prev_features[index_of_replaced_feature]}')

# UNCOMMENT BELOW IF YOU WANT TO SAVE A NEW CONFUSION MATRIX AND DISTRIBUTION PLOT
#plot_histogram(df_30s, test_labels, new_feature, -0.01, 0.15, 15, bins=30, title="Distribution of RMSE variance", task="task_3")
#save_confusion_matrix(Y_test_30s, predictions, labels, f"Task 3 - Confusion matrix\n Error rate: {1 - accuracy:.2f}", "task_3_cm")


# Task 4
features = ['zero_cross_rate_mean', 'rmse_var', 'spectral_rolloff_mean', 'chroma_stft_2_mean', 'chroma_stft_8_mean', 'mfcc_5_std'] # 6 Features that make QDA classify with 65% accuracy
#features = ['zero_cross_rate_mean', 'rmse_var', 'spectral_rolloff_mean', 'chroma_stft_5_mean', 'chroma_stft_8_mean', 'chroma_stft_5_std', 'mfcc_1_mean', 'mfcc_3_std', 'mfcc_5_std'] # 9 features that make QDA classify with 70% accuracy
#features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo'] + remaining_features # All features that make LDA classify with 71% accuracy

#df = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo'] + ['Type', 'GenreID', 'Genre'] + remaining_features)
df = pd.read_csv('data/GenreClassData_30s.txt', sep='\t', usecols=features + ['Type', 'GenreID', 'Genre'])

df_X_train = df[df['Type'] == 'Train'][features]
df_Y_train = df[df['Type'] == 'Train']['GenreID']
df_X_test  = df[df['Type'] == 'Test'][features]
df_Y_test  = df[df['Type'] == 'Test']['GenreID']

X_train = df_X_train.to_numpy()
Y_train = df_Y_train.to_numpy()
X_test  = df_X_test.to_numpy()
Y_test  = df_Y_test.to_numpy()

scaler = MaxMinScaler(feature_range=(0, 1))

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test  = scaler.transform(X_test)

classifier_flag = "LDA"

if classifier_flag == "kNN":
    classifier = kNNClassifier(k=5)
    classifier.fit(scaled_X_train, Y_train)
    predictions = classifier.predict(scaled_X_test)
elif classifier_flag == "QDA":
    classifier = QDAClassifier()
    classifier.fit(scaled_X_train, Y_train)
    predictions = classifier.predict(scaled_X_test)
elif classifier_flag == "LDA":
    classifier = LDACLassifier()
    classifier.fit(scaled_X_train, Y_train)
    predictions = classifier.predict(scaled_X_test)
else:
    accuracy = 0

accuracy = np.mean(predictions == Y_test)
print(f'Error rate with new features and {classifier_flag} classifier: {1 - accuracy:.2f}')

#save_confusion_matrix(Y_test, predictions, labels, f"Task 4 - Confusion matrix\n Error rate: {1 - accuracy:.2f}", "task_4_cm")