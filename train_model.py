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
#processed_data = pd.read_csv(data_folder + 'train_features.csv', index_col = 'id')
"""
machinelearningmastery.com/defelop-first-xgboost-model-python-scikit-learn/ 
"""
#processed_data.columns

def print_tokeninfo(sentence):
    for word in sentence:
        print ("""
text : {}
lemma : {}
dep : {}
pos :  {}
head : {}""".format(word.text,word.lemma_,word.dep_,word.pos_,word.head))



def complex_subject(subjects):
    return len(subjects) > 1

def post_processing(row):
    row['complex'] = complex_subject(row['subject_similarity_list'])    
    row['average_similarity'] = average(row['similarity_list'] )
    row['average_subject_similarity'] = average(row['subject_similarity_list'])
    row['average_verb_similarity'] = average(row['verb_similarity_list'])
    return row

def subjectmatch (word1, word2):
    return word1.dep == nsubj and word2.dep == nsubj 

def clean_lists(item):
    try:
        if isinstance(item,str):
            return 0
        elif len(item) == 0 and isinstance(item,list):
            return np.nan
        else:
            return item
    except:
        return item 

def is_subject(word):
    return word.dep == nsubj #and word.head.pos == VERB 
    
def similarity_features(q1,q2):        
    results = []    
    subject_results = [] 
    verb_results = []
    q1_subjects = []
    q2_subjects = []
    lemma_match = 0
    subject_lemma_match = 0
    verb_lemma_match = 0
    for word1 in q1:
        if is_subject(word1): 
            q1_subjects.append(word1.text)
        for word2 in q2:
            if is_subject(word2) and word2.text not in q2_subjects:
                    q2_subjects.append(word2.text)                                        
            if check_stops(word1,word2):                
                similarity = word1.similarity(word2)
                lemma_match = lemma_match_func(word1,word2,lemma_match)                
                if subjectmatch(word1,word2):
                    subject_results.append(similarity)
                    verb_similarity = word1.head.similarity(word2.head)                
                    verb_results.append(verb_similarity)
                    subject_lemma_match = lemma_match_func(word1,word2,lemma_match)
                    verb_lemma_match = lemma_match_func(word1.head,word2.head,lemma_match)
                results.append(similarity)                
    if len(subject_results) == 0:
        subject_results = np.nan
    if len(verb_results) == 0:
        verb_results = np.nan    
    return results, subject_results, verb_results, subject_lemma_match, verb_lemma_match, lemma_match, q1_subjects, q2_subjects
#    return results, q1_subjects, q2_subjects

def features_fast(row):    
    q1 = nlp(row['question1'])
    q2 = nlp(row['question2'])    
    results, subject_results, verb_results, subject_lemma_match, verb_lemma_match, lemma_match, q1_subjects, q2_subjects = similarity_features(q1,q2)            
    row['similarity_list'] = results 
    row['subject_similarity_list'] = subject_results
    row['verb_similarity_list'] = verb_results
    row['subject_match'] = subject_lemma_match
    row['verb_match'] = verb_lemma_match
    row['matches'] = lemma_match 
    row['q1_subjects'] = q1_subjects
    row['q2_subjects'] = q2_subjects 
    return row 



#q1 = nlp(example['question1']) 
#
#q2 = nlp(example['question2']) 
#
#processed_data_clean = processed_data.loc[:,:]
#
#processed_data_clean['q1_subjects'] = processed_data.q1_subjects.apply(clean_lists)
#
#processed_data_clean['q2_subjects'] = processed_data.q2_subjects.apply(clean_lists)
#
#
#processed_data_clean = processed_data_clean.fillna(0)
#
#
#unusual = processed_data_clean.query('subject_match == 0')
#
#unusual = unusual.query('q1_subjects != 0 and q2_subjects != 0')
#
#
#check = unusual.iloc[1]
#
#q1 = nlp(check['question1'])
#q2 = nlp(check['question2'])
#
#
#print_tokeninfo(q1)
#print_tokeninfo(q2)
#
#

fixed = unusual.apply(features_fast,axis = 1)
