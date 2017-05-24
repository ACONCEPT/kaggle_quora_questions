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

def average(alist):
    return (sum(alist)/len(alist))

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

def lemma_match_f(w1,w2):
    if w1.lemma == w2.lemma:
        return True
    else:
        return False


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


def similarity_features_v2(q1, q2):
    all_sims = []
    all_match = int(0)
    
    nostop_sims = []
    nostop_pairs = []
    nostops_match = int(0)
    
    subject_sims = []
    subject_pairs = []
    subjects_match = int(0)
    
    verb_sims = []
    verb_pairs = [] 
    verbs_match = int(0)  
    
    dobj_sims = [] 
    dobj_head_sims = []
    dobj_match = int(0)
    dobj_head_match = int(0)  
    
    
    pobj_sims = []
    pobj_head_sims = []
    pobj_match = int() 
    pobj_head_match = int()
    
    
    all_pairs = []
    
    for word1 in q1:
        for word2 in q2:
            all_pairs.append((word1.text,word2.text))
            
            sim = float(word1.similarity(word2))
            match = lemma_match_f(word1,word2)            
            
            #functions for all words             
            all_sims.append(sim)            
            if match:
                all_match += 1                
                
            #all words, excluding stops
            if check_stops(word1, word2):
                nostop_sims.append(sim)
                if match:
                    nostops_match += 1
                nostop_pairs.append((q1.text,q2.text))
                
                
            #features for subjects
            if word1.dep == nsubj and word2.dep == nsubj:
                subject_sims.append(sim)
                if match:
                    subjects_match += 1           
                subject_pairs.append((q1.text,q2.text))
            
            #features for verbs
            if word1.pos == VERB and word2.pos == VERB:
                verb_sims.append(sim)
                if match:
                    verbs_match += 1
                verb_pairs.append((word1.text,word2.text))
                
            if word1.dep == dobj and word2.dep == dobj:
                dobj_sims.append(sim)
                dobj_head_sims.append(word1.head.similarity(word2.head))
                if match:
                    dobj_match += 1
                if word1.head.lemma == word2.head.lemma:
                    dobj_head_match+=1
                    
            if word1.dep == pobj and word2.dep == pobj:
                pobj_sims.append(sim)
                pobj_head_sims.append(word1.head.similarity(word2.head))
                if match:
                    pobj_match += 1
                if word1.head.lemma == word2.head.lemma:
                    pobj_head_match+=1
                    
    (avg_sim, avg_nostop_sim, avg_subj_sim, 
     avg_verb_sim, avg_dobj_sim, avg_dobj_head_sim, avg_pobj_sims ,avg_pobj_head_sims
     )= averages(all_sims, nostop_sims, subject_sims, verb_sims,
     dobj_sims, dobj_head_sims, pobj_sims, pobj_head_sims )
    return (all_match, nostops_match, subjects_match, verbs_match,
                avg_sim, avg_nostop_sim, avg_subj_sim, avg_verb_sim,
                nostop_pairs, subject_pairs, verb_pairs,
                avg_dobj_sim, avg_dobj_head_sim, dobj_match, dobj_head_match,
                avg_pobj_sims ,avg_pobj_head_sims, pobj_match, pobj_head_match)

def averages(all_sims, nostop_sims, subject_sims, verb_sims,dobj_sims, dobj_head_sims,pobj_sims, pobj_head_sims):
    try:        
        avg_sim = average(all_sims) 
    except ZeroDivisionError:
        avg_sim = 0
        
    try:        
        avg_nostop_sim = average(nostop_sims)
    except ZeroDivisionError:
        avg_nostop_sim = 0
        
    try:        
         avg_subj_sim = average(subject_sims)
    except ZeroDivisionError:
        avg_subj_sim = 0
    try:        
        avg_verb_sim = average(verb_sims)
    except ZeroDivisionError:
        avg_verb_sim = 0
    
    try:        
        avg_dobj_sims= average(dobj_sims)
    except ZeroDivisionError:    
        avg_dobj_sims = 0 
        
    try:        
        avg_dobj_head_sims = average(dobj_head_sims )
    except ZeroDivisionError:
        avg_dobj_head_sims = 0 
        
    try:        
        avg_pobj_sims= average(dobj_sims)
    except ZeroDivisionError:    
        avg_pobj_sims = 0 
        
    try:        
        avg_pobj_head_sims = average(dobj_head_sims )
    except ZeroDivisionError:
        avg_pobj_head_sims = 0 
        
    return avg_sim , avg_nostop_sim , avg_subj_sim , avg_verb_sim, avg_dobj_sims, avg_dobj_head_sims, avg_pobj_sims ,avg_pobj_head_sims
    

def features_fast_v2(row):    
    q1 = nlp(row['question1'])
    q2 = nlp(row['question2'])    
    
    (all_match, nostops_match, subjects_match, verbs_match
     , avg_sim, avg_nostop_sim, avg_subj_sim, avg_verb_sim ,     
     nostop_pairs, subject_pairs, verb_pairs, avg_dobj_sim, avg_dobj_head_sim,
     dobj_match, dobj_head_match, 
     avg_pobj_sims ,avg_pobj_head_sims, pobj_match, pobj_head_match
    )= similarity_features_v2(q1,q2)
    
    row['nostop_pairs'] = nostop_pairs
    row['subject_pairs'] = subject_pairs
    row['verb_pairs'] = verb_pairs
    row['all_match'] = all_match
    row['nostops_match'] = nostops_match
    row['subjects_match'] = subjects_match
    row['verbs_match'] = verbs_match 
    row['avg_sim'] = avg_sim
    row['avg_nostop_sim'] = avg_nostop_sim
    row['avg_subj_sim'] = avg_subj_sim
    row['avg_verb_sim'] = avg_verb_sim
    row['avg_dobj_sim'] = avg_dobj_sim
    row['avg_dobj_head_sim'] = avg_dobj_head_sim
    row['dobj_match'] = dobj_match
    row['dobj_head_match'] = dobj_head_match 
    row['avg_pobj_sims']  = avg_pobj_sims
    row['avg_pobj_head_sims']= avg_pobj_head_sims 
    row['pobj_match']  = pobj_match
    row['pobj_head_match'] = pobj_head_match 
    return row 

def main ():
    s = time.time()
    processed_data = data.apply(features_fast_v2, axis = 1 )
    e = time.time()
    process_time_1 = e - s 
    processed_data.to_csv('train_features.csv')
    return processed_data, process_time_1

def test (test_size  = 50):
    start = random.randint(0,len(data) - test_size)
    sample = data.loc[start:(start + test_size),:]
    s = time.time()
    processed_data = sample.apply(features_fast_v2, axis = 1 )
    e = time.time()
    process_time_1 = e - s 
#    processed_data.to_csv('train_features.csv')
    return processed_data, process_time_1

if __name__ == '__main__':
    processed_data, process_time = main()
    

#check = processed_data.iloc[1]
#
#
#print_tokeninfo(nlp(check['question1']))
#
#print_tokeninfo(nlp(check['question2']))
#
#doc = nlp(check['question1'])
#
#print(doc[-2])
#
#doc = nlp(check['question2'])
#
#print(doc[-2])