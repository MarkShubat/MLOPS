from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from model_preprocessing import preprocessor
from sklearn.metrics import accuracy_score
from model_preparation import pipeline

import numpy as np
import os
import pandas as pd

X_test = pd.read_csv("test/X_test.csv")
predictions = pipeline.predict(X_test)
