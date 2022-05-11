import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.preprocessors import ColumnSelector, Pipeline, CaseNormalizer
from src.models import RuleBaselineModel

pipeline = Pipeline(
    CaseNormalizer(columns=['text']),
    ColumnSelector(column='text')
)
model = RuleBaselineModel()

data = pd.read_csv('data/train.csv')
train, test = train_test_split(data, random_state=1234)

X_train, y_train = pipeline.fit_transform(train), train['target'].values
X_test, y_test = pipeline.transform(test), test['target'].values

# Training
model.train(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Assessment
naive_accuracy = sum(y_test) / len(y_test)
naive_accuracy = naive_accuracy if naive_accuracy > 0.5 else (1 - naive_accuracy)

print(f"Naive test accuracy: {naive_accuracy}")
print(f"Model test accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}")
