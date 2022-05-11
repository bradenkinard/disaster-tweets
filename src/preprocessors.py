from abc import ABC, abstractmethod

class Pipeline:
    def __init__(self, **stages):
        self.stages = stages

    def transform(self, X):
        Xc = X.copy()
        for stage in self.stages.values():
            Xc = stage.transform(Xc)
        return Xc
    
    def fit_transform(self, X):
        Xc = X.copy()
        for stage in self.stages.values():
            Xc = stage.fit_transform(Xc)
        return Xc


class BasePreprocessor(ABC):
    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class NullProcessor(BasePreprocessor):
    def transform(self, X):
        return X


class CaseNormalizer(BasePreprocessor):
    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, X):
        for column in self.columns:
            X[column] = X[column].str.lower()
        return X

class ColumnSelector(BasePreprocessor):
    def __init__(self, column: str):
        self.column = column

    def transform(self, X):
        return X[self.column]
 