import numpy as np

class MaxMinScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)

    def transform(self, data):
        scaled_data = (data - self.min_) / (self.max_ - self.min_)
        scaled_data = scaled_data * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)