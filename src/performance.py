import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.maxMinScaler import MaxMinScaler
from src.kNN_classifier import kNNClassifier
from src.QDA_classifier import QDAClassifier

def save_confusion_matrix(y_true, y_pred, labels, figure_title, png_title):
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    png_name = "results/" + png_title + ".png"

    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels)
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(figure_title)
    
    plt.savefig(png_name)

def plot_histogram(dataframe, classes, feature, xlim_bottom, xlim_top, ylim_top, bins=30, title="", task=""):
    fig, ax = plt.subplots(1, len(classes), figsize=(12, 5))
    ax[0].set_ylabel("Frequency")
    for i, genre in enumerate(classes):
        ax[i].hist(dataframe[dataframe['Genre'] == genre][feature], bins=bins, color='blue', alpha=0.7)
        ax[i].set_xlim(xlim_bottom, xlim_top)  # Set x-axis limit for better visualization
        ax[i].set_ylim(0, ylim_top)  # Set y-axis limit for better visualization
        ax[i].set_xlabel(f"{feature}")
        ax[i].set_title(genre)
        ax[i].grid(axis='y', alpha=0.75)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"results/{task}_{feature}_histogram.png")

def findBestNewFeature(remaining_features, prev_features, df_full):
    revised_features = prev_features.copy()

    best_new_features = prev_features.copy()
    best_new_feature_accuracies = np.zeros(len(prev_features))

    for j, curr_feature_to_replace in enumerate(prev_features):
        new_feature_accuracies = np.zeros(len(remaining_features))

        for i, new_feature in enumerate(remaining_features):
            features = prev_features.copy()
            features[j] = new_feature

            df_X_train = df_full[df_full['Type'] == 'Train'][features]
            df_Y_train = df_full[df_full['Type'] == 'Train']['GenreID']
            df_X_test  = df_full[df_full['Type'] == 'Test'][features]
            df_Y_test  = df_full[df_full['Type'] == 'Test']['GenreID']

            X_train = df_X_train.to_numpy()
            Y_train = df_Y_train.to_numpy()
            X_test  = df_X_test.to_numpy()
            Y_test  = df_Y_test.to_numpy()
            
            scaler = MaxMinScaler(feature_range=(0, 1))

            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)

            classifier = kNNClassifier(k=5)
            classifier.fit(scaled_X_train, Y_train)
            predictions = classifier.predict(scaled_X_test)

            accuracy = np.mean(predictions == Y_test)
            new_feature_accuracies[i] = accuracy

        best_accuracy_index = np.argmax(new_feature_accuracies)
        best_new_feature_accuracies[j] = new_feature_accuracies[best_accuracy_index]
        best_new_features[j] = remaining_features[best_accuracy_index]
    
    index_of_feature_to_replace = np.argmax(best_new_feature_accuracies)
    revised_features[index_of_feature_to_replace] = best_new_features[index_of_feature_to_replace]

    return revised_features, best_new_features[index_of_feature_to_replace], index_of_feature_to_replace

def findBestNewFeatureQDA(remaining_features, prev_features, df_full):
    revised_features = prev_features.copy()

    best_new_features = prev_features.copy()
    best_new_feature_accuracies = np.zeros(len(prev_features))

    for j, curr_feature_to_replace in enumerate(prev_features):
        new_feature_accuracies = np.zeros(len(remaining_features))

        for i, new_feature in enumerate(remaining_features):
            features = prev_features.copy()
            features[j] = new_feature

            df_X_train = df_full[df_full['Type'] == 'Train'][features]
            df_Y_train = df_full[df_full['Type'] == 'Train']['GenreID']
            df_X_test  = df_full[df_full['Type'] == 'Test'][features]
            df_Y_test  = df_full[df_full['Type'] == 'Test']['GenreID']

            X_train = df_X_train.to_numpy()
            Y_train = df_Y_train.to_numpy()
            X_test  = df_X_test.to_numpy()
            Y_test  = df_Y_test.to_numpy()
            
            scaler = MaxMinScaler(feature_range=(0, 1))

            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)

            classifier = QDAClassifier()
            classifier.fit(scaled_X_train, Y_train)
            predictions = classifier.predict(scaled_X_test)

            accuracy = np.mean(predictions == Y_test)
            new_feature_accuracies[i] = accuracy

        best_accuracy_index = np.argmax(new_feature_accuracies)
        best_new_feature_accuracies[j] = new_feature_accuracies[best_accuracy_index]
        best_new_features[j] = remaining_features[best_accuracy_index]
    
    index_of_feature_to_replace = np.argmax(best_new_feature_accuracies)
    revised_features[index_of_feature_to_replace] = best_new_features[index_of_feature_to_replace]

    return revised_features, best_new_features[index_of_feature_to_replace], index_of_feature_to_replace