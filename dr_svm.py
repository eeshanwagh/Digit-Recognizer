from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np
import csv

## Kaggle Digit Recognizer
## Using Random Forests
## Eeshan Wagh


print "Reading Data"
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

Y = data['label']
columns = data.columns.tolist()
X = data[columns[1:]]

clf = svm.SVC()
clf.fit(X, Y)