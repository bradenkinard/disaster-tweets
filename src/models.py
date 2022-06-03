from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NaiveModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_  = unique_labels(y)

        self.pred = pd.Series(y).value_counts().idxmax()
        self.pred_proba = (pd.Series(y) == self.pred).mean()
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        return [self.pred] * len(X)


class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.disaster_words = [
            "earthquake",
            "quake",
            "flood",
            "tornado",
            "hurricane",
            "storm",
            "fire",
            "tsunami",
            "volcano"
        ]

    def fit(self, X, y):
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        pattern = "|".join(self.disaster_words)
        return X.str.contains(pattern).astype(int).values
