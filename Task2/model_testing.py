from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from model_preprocessing import preprocessor
from sklearn.metrics import accuracy_score
#from model_preparation import pipeline
from joblib import load
import numpy as np
import os
import pandas as pd

pipeline = load('pipeline.pkl')
X_test = pd.read_csv("test/X_test.csv")
y_test = pd.read_csv("test/y_test.csv").astype(float)
predictions = pipeline.predict(X_test)

print('Model accuracy score:', accuracy_score(y_test, predictions))
