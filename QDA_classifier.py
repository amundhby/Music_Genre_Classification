import numpy as np

# Assumes equal prior probabilities and Gaussian distributed classdata
class QDAClassifier:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.classes = np.unique(Y)
        self.means = {}
        self.covariances = {}
        for c in self.classes:
            X_c = X[Y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.covariances[c] = np.cov(X_c, rowvar=False, ddof=0)

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = []
            for c in self.classes:
                mean = self.means[c]
                covariance = self.covariances[c]
                sign, logdet = np.linalg.slogdet(covariance)
                diff = x - mean
                quad = diff.T @ np.linalg.solve(covariance, diff)
                
                score = -0.5 * (quad + logdet)
                
                class_scores.append(score)
            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)
        return np.array(predictions)