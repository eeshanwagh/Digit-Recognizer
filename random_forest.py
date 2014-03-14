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


## Random Forest

print "Construting Random Forest"
rf = RandomForestClassifier(n_estimators=1000, n_jobs=10)
#scores = cross_val_score(rf,X,Y)
#print scores.mean()
print "Estimating Random Forest Parameters"

rf.fit(X, Y)
predictions = rf.predict(test)
output = [] 
for i in range(len(predictions)):
	output.append([i+1, predictions[i]])
print len(predictions)
np.savetxt('python_rv.csv', output, delimiter=',', fmt='%d')

print "Finished"

