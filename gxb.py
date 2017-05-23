#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:03:00 2017

@author: joe
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 01:07:01 2017

@author: joe
"""
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import os
import sys 
data_folder = os.environ['HOME']  + '/repos/qqp/'
scriptpath = os.environ['HOME']  + '/repos/qqp/'
sys.path.append(scriptpath)
from features import nlp, features_fast, print_tokeninfo, lemma_match_f, check_stops


# split data into X and y
processed_data.columns 
 
X = processed_data.loc[:,'all_match':'pobj_head_match']
Y = processed_data.loc[:,'is_duplicate']



# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


print(model)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))