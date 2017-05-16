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
from features import nlp, features_fast, print_tokeninfo
processed_data = pd.read_csv(data_folder + 'train_features.csv', index = True)
"""
machinelearningmastery.com/defelop-first-xgboost-model-python-scikit-learn/ 
"""
def average(alist):
    return (sum(alist)/len(alist))

def complex_subject(subjects):
    return len(subjects) > 1

def post_processing(row):
    row['complex'] = complex_subject(row['subject_similarity_list'])    
    row['average_similarity'] = average(row['similarity_list'] )
    row['average_subject_similarity'] = average(row['subject_similarity_list'])
    row['average_verb_similarity'] = average(row['verb_similarity_list'])
    return row

def print_tokeninfo(sentence):
    for word in sentence:
        print ("""
text : {}
lemma : {}
dep : {}
pos :  {}
head : {}""".format(word.text,word.lemma_,word.dep_,word.pos_,word.head))

view = processed_data.loc[:20,:]

example = processed_data.loc[3,:]

q1 = nlp(example['question1']) 
q2 = nlp(example['question2']) 


def subjectmatch (word1, word2):
    return word1.dep == nsubj and word1.head.pos == VERB and word2.dep == nsubj and word2.head.pos == VERB

subjects = []
print (q1)
for word in q1: 
    if word.dep == nsubj and word.head.pos == VERB:
        print (word)
        subjects.append(word)

for word in subjects:
    print (word) 
    
subjects = []
print (q2)
print_tokeninfo(q2)
for word in q2: 
    if word.dep == nsubj and word.head.pos == VERB:
        print (word)
        subjects.append(word)

for word in subjects:
    print (word) 
        
print (type(test['verb_similarity_list']))


        
print_tokeninfo(nlp(test['question1']))

print_tokeninfo(nlp(test['question2']))

    try:
        average = reduce(lambda x, y: x + y, results) /len(results)
    except:
        average = 0        
        
    try:
        if len(subject_results) > 1 and isinstance(subject_results,list): 
            iscomplex = True
            subject_average = reduce(lambda x, y: x + y, subject_results) /len(subject_results)
            verb_average = reduce(lambda x, y: x + y, verb_results) /len(verb_results)
        else:
            subject_average = np.nan
            verb_average = np.nan
    except:
        if isinstance(verb_results,list):
            pass
        else:
            average = 0  



d = data.loc[:,:]

print (data.columns)

d['subject_similarity_length'] = d.subject_similarity.apply(len)
d['verb_similarity_length'] = d.verb_similarity.apply(len)


print (d.columns)

complexones = d.query('subject_similarity_length > 1')


check = d.iloc[2]

sim = check['verb_similarity']

sim = []
print (len(sim))

for thing in sim:
    print(thing)

q1 = nlp(check['question1'])
q2 = nlp(check['question1'])


print_tokeninfo(q1)  
print_tokeninfo(q2)