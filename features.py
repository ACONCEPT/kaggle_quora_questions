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
import random
from spacy.symbols import nsubj, VERB, pobj, dobj
nlp = spacy.load('en')
#tagged = pd.read_csv(data_folder + 'tagged_train.csv')
data_folder = os.environ['HOME']  + '/repos/qqp/'
data = pd.read_csv(data_folder + 'train.csv', index_col = 'id')
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
        print ("""
text : {}
lemma : {}
dep : {}
pos :  {}
head : {}""".format(word.text,word.lemma_,word.dep_,word.pos_,word.head))
        
#        print (word.text,word.lemma_,word.dep_,word.pos_,word.head)
        
def subjectmatch (word1, word2):
    return is_subject(word1) and is_subject(word2)

def is_subject(word):
    return word.dep == nsubj and word.head.pos == VERB 

def lemma_match_func(w1,w2,i):
    if w1.lemma == w2.lemma:
        i += 1
        return  i 
    else:
        return i


def check_stops(w1,w2):
    return not w1.is_stop and not w2.is_stop
    

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
        for word2 in q2:            
            if check_stops(word1,word2):                
                similarity = word1.similarity(word2)
                lemma_match = lemma_match_func(word1,word2,lemma_match)                
                if is_subject(word1): 
                    q1_subjects.append(word1)
                if is_subject(word2):
                    q2_subjects.append(word2)                                        
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

def main ():
    s = time.time()
    processed_data = data.apply(features_fast, axis = 1 )
    e = time.time()
    process_time_1 = e - s 
    processed_data.to_csv('train_features.csv')
    return processed_data, process_time_1

def test (test_size  = 50):
    start = random.randint(0,len(data) - test_size)
    sample = data.loc[start:(start + test_size),:]
    s = time.time()
    processed_data = sample.apply(features_fast, axis = 1 )
    e = time.time()
    process_time_1 = e - s 
    processed_data.to_csv('train_features.csv')
    return processed_data, process_time_1

if __name__ == '__main__':
    processed_data, process_time = main()
    
    
    
    
