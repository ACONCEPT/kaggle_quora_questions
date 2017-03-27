# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os 
import pandas as pd
import numpy as np
import math
data_folder = os.environ['HOME']  + '/repos/qqp/'
from nltk import word_tokenize  , pos_tag
from nltk.tag import map_tag

def tag_sentence (sentence):
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    simpletag = [(word, map_tag('en-ptb','universal',tag)) for word, tag in tagged]        
    indx = range(len(words))
    simpletag = [simpletag[i][1] for i in indx]
    dtag = [tagged[i][1] for i in indx]
    return pd.DataFrame({"word":words,"simpletag":simpletag,"dtag":dtag},index = indx)

def prep_wordlist (alist):
    return [x.upper().rstrip().lstrip() for x in alist] 

def count_matching_noun_verb(q1,q2):
#    if isinstance(q1,basestring) and isinstance(q2,basestring): 
    q1_t = tag_sentence (q1)
    q2_t = tag_sentence (q2)
    q1_nouns  = q1_t.query("""simpletag == 'NOUN'""")
    q1_nouns = prep_wordlist (list(q1_nouns['word']))
    q1_verbs  = q1_t.query("""simpletag == 'VERB'""")
    q1_verbs = prep_wordlist (list(q1_verbs['word']))
    q2_nouns  = q2_t.query("""simpletag == 'NOUN'""")    
    q2_nouns = prep_wordlist (list(q2_nouns['word']))
    q2_verbs  = q2_t.query("""simpletag == 'VERB'""")    
    q2_verbs = prep_wordlist (list(q2_verbs['word']))
    matching_nouns = 0
    matching_verbs = 0    
    for word in q1_nouns:
        matching_nouns += q2_nouns.count(word)
    for word in q2_verbs:
        matching_verbs += q2_verbs.count(word)
    return matching_nouns, matching_verbs 

class all_data(object):
    def __init__(self,all_data = True,traindata = False, testdata = False,  samplesubmission = False):        
        self.data_catalog = [] 
        self.chunk_size = 20000        
        self.train_chunk = 0        
        self.test_chunk = 0
        if all_data:
            self.testdata = True
            self.traindata = True
            self.samplesubmission = True
        else: 
            self.traindata = traindata
            self.testdata = testdata
            self.samplesubmission = samplesubmission
        if testdata:
#            print ('adding test data')
            self.test_data = pd.read_csv(data_folder + 'test.csv')
            self.data_catalog.append('test_data')
            self.max_test_chunks = math.ceil(len(self.test_data)/self.chunk_size)        
        if traindata:
#            print ('adding train data')
            self.train_data = pd.read_csv(data_folder + 'train.csv')
            self.data_catalog.append('train_data')
            self.max_train_chunks = math.ceil(len(self.train_data)/self.chunk_size)        
            self.train_data_features = pd.DataFrame()
        if samplesubmission:
            self.sample_submission = pd.read_csv(data_folder + 'sample_submission.csv')
            
    def change_chunk_size (self, newsize):
        self.chunk_size = newsize
        if self.traindata:
#            print (' max_train_chunks updated to {} / {}'.format(len(self.train_data),self.chunk_size))
            self.max_train_chunks = math.ceil(len(self.train_data)/self.chunk_size)            
        if self.testdata:            
            self.max_test_chunks = math.ceil(len(self.test_data)/self.chunk_size)
        return None              
    
    def reset_train_chunks(self):
        self.train_chunk = 0
        return None
        
    def rests_test_chunks(self):
        self.test_chunk = 0
        return None
            
    def get_train_data_chunk(self):        
        if self.train_chunk > self.max_train_chunks:
            print('with a chunk size of {0} maxvalue for n is {1}, {2} was requested'.format(chunk_size,max_chunks,n))
            raise TypeError    
        beginning = self.train_chunk * self.chunk_size 
        end = ((self.train_chunk+1) * self.chunk_size) - 1         
        self.train_chunk += 1                 
        return self.train_data.loc[beginning:end]        
    
    def process_training_feature_chunk (self):                    
        chunk = self.get_train_data_chunk()        
        trainit = pd.DataFrame()
        errors = pd.DataFrame()        
        columns = ['noun_match','verb_match','isdup']
        for i, chunkrow in chunk.iterrows():
            try:
                q1 = chunkrow['question1']
                q2 = chunkrow['question2']
                isdup = chunkrow['is_duplicate']   
                nmatch, vmatch = count_matching_noun_verb(q1,q2)
                muhdata = [nmatch,vmatch,isdup]
                indx = [i]
                new_item = pd.DataFrame(dict(zip(columns,muhdata)),index = indx)
                trainit = trainit.append(new_item)
            except Exception as e:
                chunkrow['error'] = e 
                new_item = pd.DataFrame(chunkrow,index = indx)        
                errors = errors.append(new_item)                
#        print ("saving data and errors to {0}".format(self.train_chunk))
#        trainit.to_csv(data_folder + 'chunked_trained_data/train_chunk_{0}'.format(self.train_chunk))
#        errors.to_csv(data_folder + 'chunked_trained_data/errors_chunk_{0}'.format(self.train_chunk))
        return trainit, errors
    
    def process_training_feature_chunk_v2 (self):            
        chunk = self.get_train_data_chunk()
        trainit = pd.DataFrame()
        errors = pd.DataFrame()        
        columns = ['noun_match','verb_match','isdup']
        for i, chunkrow in chunk.iterrows():
            try:
                q1 = chunkrow['question1']
                q2 = chunkrow['question2']
                isdup = chunkrow['is_duplicate']   
                nmatch, vmatch = count_matching_noun_verb(q1,q2)
                chunkrow['noun_match'] = nmatch
                chunkrow['verb_match'] = vmatch      
                indx = [i]
                new_item = pd.DataFrame(chunkrow.to_dict(),indx)
                trainit = trainit.append(new_item)
            except Exception as e:
                chunkrow['error'] = e 
                new_item = pd.DataFrame(chunkrow,index = indx)        
                errors = errors.append(new_item)                
#        print ("saving data and errors to {0}".format(self.train_chunk))
#        trainit.to_csv(data_folder + 'chunked_trained_data/train_chunk_{0}_v2'.format(self.train_chunk))
#        errors.to_csv(data_folder + 'chunked_trained_data/errors_chunk_{0}_v2'.format(self.train_chunk))
        return trainit, errors
