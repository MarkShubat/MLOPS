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

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print('Model accuracy score:', accuracy_score(y_valid, preds))
