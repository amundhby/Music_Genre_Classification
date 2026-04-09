import numpy as np
import pandas as pd

class kNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predicted_labels = [self._predict(x) for x in X_test]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute distances between x and all samples in the training set
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label among the neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common