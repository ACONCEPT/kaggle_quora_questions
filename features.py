# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import nltk
#from nltk.tag.perceptron import PerceptronTagger
#from nltk.tag import map_tag, pos_tag
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()

import sys
import os 
import pandas as pd
import numpy as np
import math
import time
import spacy
from spacy.symbols import nsubj, VERB, pobj, dobj
nlp = spacy.load('en')
#tagged = pd.read_csv(data_folder + 'tagged_train.csv')
data_folder = os.environ['HOME']  + '/repos/qqp/'
data = pd.read_csv(data_folder + 'train.csv')
data ['question1'] = data.question1.apply(str)
data ['question2'] = data.question2.apply(str)
from itertools import combinations, chain
from functools import reduce

def tag_question (q):
    return nlp(q)

def get_subjects(q):
    subjects = []
    for word in q:
        if word.dep == nsubj and word.head.pos == VERB:
            subjects.append(word)
    return subjects        

def print_tokeninfo(sentence):
    for word in sentence:
        print (word.text,word.lemma_,word.dep_,word.pos_,word.head)
        
def subjectmatch (word1, word2):
    return word1.dep == nsubj and word1.head.pos == VERB and word2.dep == nsubj and word2.head.pos == VERB

def similarity_features(q1,q2):        
    results = []    
    verb_results = []
    subject_results = []
    for word1 in q1:        
        for word2 in q2:                            
            similarity = word1.similarity(word2)
            if subjectmatch(word1,word2):
                subject_results.append(similarity)
                verb_similarity = word1.head.similarity(word2.head)
                verb_results.append(verb_similarity)
            results.append(similarity)
    average = reduce(lambda x, y: x + y, results) /len(results)
    return average, subject_results, verb_results
#    return results, q1_subjects, q2_subjects

def average_similarity(q1,q2):    
    similarity_tuple = () 
    results = []
    i = 0 
    for word1 in q1:        
        for word2 in q2:            
            similarity = word1.similarity(word2)
            results.append(similarity)
#            similarity_tuple = (word1.text,word2.tex0t,similarity)
#            results.append(similarity_tuple)            
    return reduce(lambda x, y: x + y, results) /len(results)

def subject_match_vector(q1,q2):
    q1_subjects = get_subjects(q1)
    q2_subjects = get_subjects(q2)
    similarity_tuple = () 
    results = []
    for subject1 in q1_subjects:
#        print(subject1)
        for subject2 in q2_subjects:
#            print(subject2)
            similarity = subject1.similarity(subject2)
            similarity_tuple = (subject1.text,subject2.text,similarity)
            results.append(similarity)
#            results.append(similarity_tuple)
    return results, q1_subjects, q2_subjects

def verb_match_vector(q1_subjects,q2_subjects):
    q1_verbs = [subject.head for subject in q1_subjects]
    q2_verbs = [subject.head for subject in q2_subjects]    
    similarity_tuple = () 
    results = []
    for verb1 in q1_verbs:
        for verb2 in q2_verbs:
            similarity = verb1.similarity(verb2)
            similarity_tuple = (verb1.text,verb2.text,similarity)
            results.append(similarity)
#            results.append(similarity_tuple)
    return results#, q1_verbs, q2_verbs

def features(row):
    q1 = nlp(row['question1'])
    q2 = nlp(row['question2'])
    row['subject_match' ], q1_subjects,q2_subjects = subject_match_vector(q1,q2)
    row['verb_match' ] = verb_match_vector(q1_subjects,q2_subjects)    
    row['average_similarity'] =  average_similarity(q1,q2)
    return row

def features_fast(row):    
    q1 = nlp(row['question1'])
    q2 = nlp(row['question2'])    
    average, subject_results, verb_results = similarity_features(q1,q2)
    row['average_similarity'] = average
    row['subject_similarity'] = subject_results
    row['verb_similarity'] = verb_results
    return row 

s = time.time()
processed_data = data.apply(features_fast, axis = 1 )
e = time.time()
process_time_1 = e - s 


s = time.time()
processed_data = data.apply(features, axis = 1 )
s = time.time()
process_time_2 = e - s 

processed_data.to_csv('train_w_features.csv')