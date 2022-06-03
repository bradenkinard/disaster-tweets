from abc import ABC, abstractmethod
import re


class BasePreprocessor(ABC):
    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class DocCleaner(BasePreprocessor):
    def transform(self, docs):
        return docs.apply(clean_doc)


def clean_doc(doc):
    doc = doc.lower()
    doc = re.sub(r"[^\w\s]+", "", doc)
    doc = re.sub(r"\b[0-9]+\b", "", doc)
    doc = re.sub(r"\s+", " ", doc)

    return doc
