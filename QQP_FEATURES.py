# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os 
import pandas as pd
import math
data_folder = os.environ['HOME']  + '/repos/qqp/'

class all_data(object):
    def __init__(self,all_data = True,testdata = False, traindata = False, samplesubmission = False):        
        self.data_catalog = [] 
        self.train_chunk = 0
        self.test_chunk = 0        
        if all_data:
            testdata = True
            traindata = True
            samplesubmission = True
        if testdata:
            self.test_data = pd.read_csv(data_folder + 'test.csv')
            self.data_catalog.append('test_data')
        if traindata:
            self.train_data = pd.read_csv(data_folder + 'train.csv')
            self.data_catalog.append('train_data')
        if samplesubmission:
            self.sample_submission = pd.read_csv(data_folder + 'sample_submission.csv')              


try: 
    data
except NameError:
    data = all_data()
    
try:
    train_data
except NameError:
    train_data = data.train_data

def get_train_data_chunk(n = 0, chunk_size = 100000):
    max_chunks = math.ceil(len(train_data)/chunk_size)
    if n > max_chunks:
        print('with a chunk size of {0} maxvalue for n is {1}, {2} was requested'.format(chunk_size,max_chunnks,n))
        raise TypeError    
    beginning = n * chunk_size 
    end = (n+1) * chunk_size
#    return beginning, end
    return train_data.loc[beginning:end]        

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

        
def process_feature_chunk ():
    chunk = get_train_data_chunk(data.train_chunk)
    trainit = pd.DataFrame()
    errors = pd.DataFrame()
    for i in range(len(chunk)):
        try:
            q1 = train_data.iloc[i]['question1']
            q2 = train_data.iloc[i]['question2']
            isdup = train_data.iloc[i]['is_duplicate']   
            indx = [i]
            nmatch, vmatch = count_matching_noun_verb(q1,q2)    
            columns = ['noun_match','verb_match','isdup']
            muhdata = [nmatch,vmatch,isdup]
            new_item = pd.DataFrame(dict(zip(columns,muhdata)),index = indx)
            trainit = trainit.append(new_item)
            data.train_chunk += 1 
        except Exception as e: 
            new_item = pd.DataFrame(dict(zip(columns,muhdata)),index = indx)        
            errors = errors.append(new_item)
    return trainit, errors 