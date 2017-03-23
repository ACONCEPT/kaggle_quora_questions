# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os 
import pandas as pd
data_folder = os.environ['HOME']  + '/repos/qqp/'

#for thing, row in os.environ.items():        
#    print ('{0}  : {1} '.format(thing, row)) 
#    


#for file in os.listdir(data_folder):
#    print ("self.{0} = pd.read_csv(data_folder + '{1}')".format(file.rstrip('.csv'),file))      
class all_data(object):
    def __init__(self):        
        self.test = pd.read_csv(data_folder + 'test.csv')
        self.train = pd.read_csv(data_folder + 'train.csv')
        self.sample_submission = pd.read_csv(data_folder + 'sample_submission.csv')
        
data = all_data()        
#test_data = data.test
train_data = data.train
#sample_submission = data.sample_submission

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
        
trainit = pd.DataFrame()
for i in range(len(train_data)):
    q1 = train_data.iloc[i]['question1']
    q2 = train_data.iloc[i]['question2']
    isdup = train_data.iloc[i]['is_duplicate']   
    indx = [i]
    nmatch, vmatch = count_matching_noun_verb(q1,q2)    
    columns = ['noun_match','verb_match','isdup']
    muhdata = [nmatch,vmatch,isdup]
    new_item = pd.DataFrame(dict(zip(columns,muhdata)),index = indx)
    trainit = trainit.append(new_item)





#
#for i in range(len(one_question_pair)):
#    q1 = train_data.iloc[i]['question1']
#    q2 = train_data.iloc[i]['question2']
#    q1_t = tag_sentence (q1)
#    q2_t = tag_sentence (q2)
#    q1_nouns  = q1_t.query("""simpletag == 'NOUN'""")
#    q1_nouns = prep_wordlist (list(q1_nouns['word']))
#    q1_verbs  = q1_t.query("""simpletag == 'VERB'""")
#    q1_verbs = prep_wordlist (list(q1_verbs['word']))
#    q2_nouns  = q2_t.query("""simpletag == 'NOUN'""")    
#    q2_nouns = prep_wordlist (list(q2_nouns['word']))
#    q2_verbs  = q2_t.query("""simpletag == 'VERB'""")    
#    q2_verbs = prep_wordlist (list(q2_verbs['word']))
#    matching_nouns = 0
#    matching_verbs = 0    
#    for word in q1_nouns:
#        matching_nouns += q2_nouns.count(word)
#    for word in q2_verbs:
#        matching_verbs += q2_verbs.count(word)
#    
#    
#    
#print (q1_verbs)    
#print (q2_verbs)
#
#print (q1_nouns)    
#print (q2_nouns)
#    
#astring = ' ass ' 
#astring = astring.ltrim()
#        
#    
#    
#    
#    
#
#
#
##q_1_word_tokenized = word_tokenize(question1)
##q1word1 = q_1_word_tokenized[0]
##q_1_tagged = pos_tag (q_1_word_tokenized)
##q1word1 = q_1_tagged[0][1]
##simplifiedtag = [(word, map_tag('en-ptb','universal',tag)) for word, tag in q_1_tagged]
##
##doubletag = pd_DataFrame ({d})
##
##print ( simplifiedtag )
##
##q_2_word_tokenized = word_tokenize(question2)
#
#
#
#
#
#one_question_pair = train_data.iloc[0]
#question1 = one_question_pair['question1']
#question2 = one_question_pair['question2']
#tagged_question_1 = tag_sentence(question1)
