from abc import ABC, abstractmethod
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


class NaiveModel(BaseModel):
    def train(self, X, y):
        self.pred = pd.Series(y).value_counts().idxmax()
        self.pred_proba = (pd.Series(y) == self.pred).mean()
        return self
    
    def predict(self, X):
        return [self.pred] * len(X)
    
    def predict_proba(self, X):
        return [self.pred_proba] * len(X)


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
    
    def predict_proba(self, X):
        return [1] * len(X) 


class NaiveBayes(BaseModel):
    def __init__(self):
        self.nb = MultinomialNB()
    
    def train(self, X, y):
        self.nb.fit(X, y)
        return self
    
    def predict(self, X):
        return self.nb.predict(X)
    
    def predict_proba(self, X):
        return self.nb.predict_proba(X)[:, 1]
