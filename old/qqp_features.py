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
import time
data_folder = os.environ['HOME']  + '/repos/qqp/'
from nltk.tag import map_tag, pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
from nltk.tag.perceptron import PerceptronTagger
import spacy
#nltk.download() 
nlp = spacy.load('en')
tagged = pd.read_csv(data_folder + 'tagged_train.csv')
"""
trying to find faster way to do pos_tagging.
"""


def wordlist(sentence):
    return prep_wordlist(word_tokenize(sentence))

def add_features(df):
    log = {}
    s = time.time()
    df['q1_words'] = df.question1.apply(wordlist)
    e = time.time()
    log['q1_wordlist_time'] = e - s    
    s = time.time()
    df['q1_tag'] = df.q1_words.apply(tagger.tag)
#    df['q1_tag'] = df.q1_words.apply(pos_tag)
    e = time.time()
    log['q1_tag_time'] = e - s    
    s = time.time() 
    df['q1_simpletag'] = df.q1_tag.apply(lambda s:[(word, map_tag('en-ptb','universal',tag)) for word, tag in s])
    e = time.time() 
    log['q1_simpletag_time'] = e - s    
    return df, log 

def wordlist(sentence):
    return prep_wordlist(word_tokenize(sentence))


def add_features(df):
    log = {}
    s = time.time()
    df['q1_words'] = df.question1.apply(wordlist)
    e = time.time()
    return df, log 

item = tagged.loc[1,:]

q1 = item['question1']
q2 = item['question2']

isdup = item['is_duplicate']

q1_doc = nlp(q1)
q2_doc = nlp(q2)

a = q1_doc[0].text
b = q1_doc[0].text

b = q1_doc[0].vector.shape

spacy


a = q1_doc[0].dep_
b = q1_doc[0].dep_

for q1_word, q2_word in zip ( q1_doc, q2_doc):
    print ( q1_word.text, q1_word.lemma_, q1_word.tag_,q1_word.dep_)
    print ( q2_word.text, q2_word.lemma_, q2_word.tag_, q2_word.dep_)
    
    
for q1_word in q1_doc:
    print ( q1_word.text, q1_word.lemma_, q1_word.tag_,q1_word.dep_)
    
for q1_word in q2_doc:
    print ( q1_word.text, q1_word.lemma_, q1_word.tag_,q1_word.dep_)
    


    
verbs = set()
for possible_subject in q1_doc:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        print (possible_subject,possible_subject.head)
        verbs.add(possible_subject.head)
        
for word in q1_doc:
    print (word, word.dep_)
    if word.dep == pobj:
        print ('primary object is {} '.format(word.text, word.head))
        
def get_chunk_dict(q):    
    result = {}
#    pobjs = []
#    dobjs = []         
    for tok in q:
#        print (' token = {}, dep = {} , head = {}'.format(tok.text, tok.dep_, tok.head))
        if tok.dep == nsubj and tok.head.pos == VERB:
            result[tok.dep_] = tok
    return result
#        elif tok.dep  == pobj:
#            pobjs.append(tok)
#        elif tok.dep == dobj:
#            dobjs.append(tok)

def matching_subject_lemma(q1_toks, q2_toks):
    
    

q1_d = dict(get_chunk_dict(q1_doc))
q2_d = dict(get_chunk_dict(q2_doc))


children = q1_d['nsubj'].children

for child in children:
    print(child,child.pos_,child.dep_)

print(children) 

print (q1_d['nsubj'].children)

print (q1_d['pobj'].head.dep_)

print ()

for word in q1_d['nsubj'].subtree:
    print (word, word.dep_, word.head) 

print (q2_d)

print (type(VERB))
            
for word in doc:
    print(word.dep)
#    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

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
            self.test_data = pd.read_csv(data_folder + 'test.csv')
            self.data_catalog.append('test_data')
            self.max_test_chunks = math.ceil(len(self.test_data)/self.chunk_size)        
        if traindata:
#            print ('adding train data')
            self.train_data = pd.read_csv(data_folder + 'train.csv') #404290 rows
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
        # get a chunk from the all_data object
        chunk = self.get_train_data_chunk()        
        #initialize dataframes
        trainit = pd.DataFrame()
        errors = pd.DataFrame()        
        # column names variable 
        columns = ['noun_match','verb_match','isdup']
        for i, chunkrow in chunk.iterrows():
            try:
                #split the questions apart
                q1 = chunkrow['question1']
                q2 = chunkrow['question2']
                #label of the training data 
                isdup = chunkrow['is_duplicate']   
                #2 features, noun match, verb match
                nmatch, vmatch = count_matching_noun_verb(q1,q2)
                # features and label
                muhdata = [nmatch,vmatch,isdup]
                indx = [i]
                #processed dataframe
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
            indx = [i]
            try:
                q1 = chunkrow['question1']
                q2 = chunkrow['question2']
                isdup = chunkrow['is_duplicate']   
                nmatch, vmatch = count_matching_noun_verb(q1,q2)
                chunkrow['noun_match'] = nmatch
                chunkrow['verb_match'] = vmatch                      
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
    
    def process_training_feature_chunk_v3 (self):
        chunk = self.get_train_data_chunk()
        trainit = pd.DataFrame()
        errors = pd.DataFrame()
        columns = ['noun_match','verb_match','isdup']
        for i, chunkrow in chunk.iterrows():
            indx = [i]
            try:
                q1 = chunkrow['question1']
                q2 = chunkrow['question2']
                isdup = chunkrow['is_duplicate']
                nmatch, vmatch = count_matching_noun_verb(q1,q2)
                nmatch_lem, vmatch_lem  = count_matching_noun_verb_lemma(q1,q2)
                chunkrow['noun_match'] = nmatch
                chunkrow['verb_match'] = vmatch
#                chunkrow['noun_match_lem'] = nmatch_lem
#                chunkrow['verb_match_lem'] = vmatch_lem                  
                new_item = pd.DataFrame(chunkrow.to_dict(),indx)
                trainit = trainit.append(new_item)
                if i % 10000 == 0: 
                    print('{} rows processed'.format(i))
            except Exception as e:
                chunkrow['error'] = e 
                new_item = pd.DataFrame(chunkrow,index = indx)        
                errors = errors.append(new_item)                
#        print ("saving data and errors to {0}".format(self.train_chunk))
#        trainit.to_csv(data_folder + 'chunked_trained_data/train_chunk_{0}_v2'.format(self.train_chunk))
#        errors.to_csv(data_folder + 'chunked_trained_data/errors_chunk_{0}_v2'.format(self.train_chunk))
        return trainit, errors

def tag_sentence (sentence):
    words = word_tokenize(sentence)
    words = prep_wordlist(words)
    tagged = pos_tag(words)
    simpletag = [(word, map_tag('en-ptb','universal',tag)) for word, tag in tagged]        
    indx = range(len(words))
    simpletag = [simpletag[i][1] for i in indx]
    dtag = [tagged[i][1] for i in indx]
    return words, tagged, simpletag   

    
#    return pd.DataFrame({"word":words,"simpletag":simpletag,"dtag":dtag},index = indx)

def prep_wordlist (alist):
    return [x.upper().rstrip().lstrip() for x in alist] 

""" 
all of these features take tagged sentence as input because they all need tagged
input and its faster to just do it once from the main routine
""" 
def count_matching_noun_verb(q1_t,q2_t):
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

def count_matching_noun_verb_lemma(q1_t,q2_t):
#    if isinstance(q1,basestring) and isinstance(q2,basestring): 
    q1_t = tag_sentence (q1)
    q2_t = tag_sentence (q2)
    q1_nouns  = q1_t.query("""simpletag == 'NOUN'""")
    q1_nouns = prep_wordlist (list(q1_nouns['word']))
    q1_nouns_lemma = [lemmatizer.lemmatize(word) for word in q1_nouns]
    q1_verbs  = q1_t.query("""simpletag == 'VERB'""")
    q1_verbs = prep_wordlist (list(q1_verbs['word']))
    q1_verbs_lemma = [lemmatizer.lemmatize(word) for word in q1_verbs]
    q2_nouns  = q2_t.query("""simpletag == 'NOUN'""")    
    q2_nouns = prep_wordlist (list(q2_nouns['word']))
    q2_nouns_lemma = [lemmatizer.lemmatize(word) for word in q2_nouns]
    q2_verbs  = q2_t.query("""simpletag == 'VERB'""")    
    q2_verbs = prep_wordlist (list(q2_verbs['word']))
    q2_verbs_lemma = [lemmatizer.lemmatize(word) for word in q2_verbs]
    matching_nouns_lemma = 0
    matching_verbs_lemma = 0    
    for word in q1_nouns_lemma:
        matching_nouns_lemma += q2_nouns_lemma.count(word)
    for word in q1_verbs_lemma:
        matching_verbs_lemma += q2_verbs_lemma.count(word)
    return matching_nouns_lemma, matching_verbs_lemma

#def word_similarity(w1,w2):
##    if isinstance(q1,basestring) and isinstance(q2,basestring): 
#    q1_t = tag_sentence (q1)
#    q2_t = tag_sentence (q2)
#    q1_nouns  = q1_t.query("""simpletag == 'NOUN'""")
#    q1_nouns = prep_wordlist (list(q1_nouns['word']))
#    q1_nouns_lemma = [lemmatizer.lemmatize(word) for word in q1_nouns]
#    q1_verbs  = q1_t.query("""simpletag == 'VERB'""")
#    q1_verbs = prep_wordlist (list(q1_verbs['word']))
#    q1_verbs_lemma = [lemmatizer.lemmatize(word) for word in q1_verbs]
#    q2_nouns  = q2_t.query("""simpletag == 'NOUN'""")    
#    q2_nouns = prep_wordlist (list(q2_nouns['word']))
#    q2_nouns_lemma = [lemmatizer.lemmatize(word) for word in q2_nouns]
#    q2_verbs  = q2_t.query("""simpletag == 'VERB'""")    
#    q2_verbs = prep_wordlist (list(q2_verbs['word']))
#    q2_verbs_lemma = [lemmatizer.lemmatize(word) for word in q2_verbs]
#    matching_nouns_lemma = 0
#    matching_verbs_lemma = 0    
#    for word in q1_nouns_lemma:
#        matching_nouns_lemma += q2_nouns_lemma.count(word)
#    for word in q1_verbs_lemma:
#        matching_verbs_lemma += q2_verbs_lemma.count(word)
#    return matching_nouns_lemma, matching_verbs_lemma

def main():
    data = all_data(False,True)
    data.change_chunk_size(100)
    data.reset_train_chunks()
    chunk = data.get_train_data_chunk()
    q1 = chunk.iloc[0]['question1']
    q2 = chunk.iloc[0]['question2']
    isdup = chunk.iloc[0]['is_duplicate']
    q1_t = tag_sentence (q1)
    q2_t = tag_sentence (q2)
    lemma_noun_match , lemma_verb_match = count_matching_noun_verb_lemma(q1_t,q2_t)    
    print ("lemma noun match : {}, lemma verb match : {}".format(lemma_noun_match, lemma_verb_match))
    return q1, q2, lemma_noun_match, lemma_verb_match, isdup, data 

def get_a_chunk(size):
    data = all_data(False,True)
    data.change_chunk_size(size)
    data.reset_train_chunks()
    chunk = data.get_train_data_chunk()
    return chunk

def get_full_data():
    data = all_data(False,True)
    return data
#    print ("lemma noun match : {}, lemma verb match : {}".format(lemma_noun_match, lemma_verb_match))
#    return q1, q2, lemma_noun_match, lemma_verb_match, isdup
if __name__ == "__main__":
    full_data = get_full_data()
#    q1, q2, lemma_noun_match, lemma_verb_match, isdup , data = main ()
#    chunk = get_a_chunk(10000)


tagger = PerceptronTagger()




featured_chunk, runinfo = add_features(full_data.train_data)

featured_chunk.to_csv('tagged_train.csv')



#data_read = chunk.train_data
#row = data_read[:3]
#