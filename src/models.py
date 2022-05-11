from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class NaiveModel(BaseModel):
    def train(self, X, y):
        self.pred = pd.Series(y).value_counts().idxmax()
        return self
    
    def predict(self, X):
        return [self.pred] * len(X)


class HeuristicModel(BaseModel):
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

    def train(self, X, y):
        return self
    
    def predict(self, X):
        pattern = "|".join(self.disaster_words)
        return X.str.contains(pattern).astype(int).values
