from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from model_preprocessing import preprocessor
from sklearn.metrics import accuracy_score

import numpy as np
import os
import pandas as pd


model = RandomForestClassifier(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])
df = pd.read_csv("titanic.csv")

X_train = pd.read_csv("train/X_train.csv")
y_train = pd.read_csv("train/y_train.csv")["Survived"]
X_valid = pd.read_csv("test/X_test.csv")
y_valid = pd.read_csv("test/y_test.csv")["Survived"]
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print('Model accuracy score:', accuracy_score(y_valid, preds))
