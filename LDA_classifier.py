import numpy as np

# Assumes equal prior probabilities and gaussian distributed classdata with equal covariance matrices
class LDACLassifier:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.classes = np.unique(Y)
        self.means = {}
        self.covariance = np.cov(X, rowvar=False, ddof=0)
        for c in self.classes:
            X_c = X[Y == c]
            self.means[c] = np.mean(X_c, axis=0)

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = []
            for c in self.classes:
                mean = self.means[c]
                w_c = np.linalg.solve(self.covariance, mean)
                b_c = -0.5 * mean.T @ np.linalg.solve(self.covariance, mean)
                
                score = w_c.T @ x + b_c
                
                class_scores.append(score)
            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)
        return np.array(predictions)