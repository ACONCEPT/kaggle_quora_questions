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
        print ("""
text : {}
lemma : {}
dep : {}
pos :  {}
head : {}""".format(word.text,word.lemma_,word.dep_,word.pos_,word.head))
        
#        print (word.text,word.lemma_,word.dep_,word.pos_,word.head)
        
def subjectmatch (word1, word2):
    return word1.dep == nsubj and word1.head.pos == VERB and word2.dep == nsubj and word2.head.pos == VERB

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
    lemma_match = 0
    subject_lemma_match = 0
    verb_lemma_match = 0
    for word1 in q1:        
        for word2 in q2:
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
    return results, subject_results, verb_results, subject_lemma_match, verb_lemma_match, lemma_match
#    return results, q1_subjects, q2_subjects

def features_fast(row):    
    q1 = nlp(row['question1'])
    q2 = nlp(row['question2'])    
    results, subject_results, verb_results, subject_lemma_match, verb_lemma_match, lemma_match = similarity_features(q1,q2)            
    row['similarity_list'] = results 
    row['subject_similarity_list'] = subject_results
    row['verb_similarity_list'] = verb_results
    row['subject_match'] = subject_lemma_match
    row['verb_match'] = verb_lemma_match
    row['matches'] = lemma_match 
    return row 

def main ():
    s = time.time()
    processed_data = data.apply(features_fast, axis = 1 )
    e = time.time()
    process_time_1 = e - s 
    processed_data.to_csv('train_features.csv')
    return processed_data, process_time_1

if __name__ == '__main__':
    processed_data, process_time = main()


#
#def verb_match_vector(q1_subjects,q2_subjects):
#    q1_verbs = [subject.head for subject in q1_subjects]
#    q2_verbs = [subject.head for subject in q2_subjects]    
#    similarity_tuple = () 
#    results = []
#    for verb1 in q1_verbs:
#        for verb2 in q2_verbs:
#            similarity = verb1.similarity(verb2)
##            similarity_tuple = (verb1.text,verb2.text,similarity)
#            results.append(similarity)
##            results.append(similarity_tuple)
#    return results#, q1_verbs, q2_verbs
#

#
#def subject_match_vector(q1,q2):
#    q1_subjects = get_subjects(q1)
#    q2_subjects = get_subjects(q2)
#    similarity_tuple = () 
#    results = []
#    for subject1 in q1_subjects:
##        print(subject1)
#        for subject2 in q2_subjects:
##            print(subject2)
#            similarity = subject1.similarity(subject2)
#            similarity_tuple = (subject1.text,subject2.text,similarity)
#            results.append(similarity)
##            results.append(similarity_tuple)
#    return results, q1_subjects, q2_subjects
#
#
#def features(row):
#    q1 = nlp(row['question1'])
#    q2 = nlp(row['question2'])    
#    row['subject_match' ], q1_subjects,q2_subjects = subject_match_vector(q1,q2)
#    row['verb_match' ] = verb_match_vector(q1_subjects,q2_subjects)    
#    row['average_similarity'] =  average_similarity(q1,q2)
#    return row