# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
data_folder = '/home/kaggle/anaconda3/QUESTIONPAIRS/'
import sys
import os 
import pandas as pd

#for file in os.listdir(data_folder):
#    print ("self.{0} = pd.read_csv(data_folder + '{1}')".format(file.rstrip('.csv'),file))      
class all_data(object):
    def __init__(self):        
        self.test = pd.read_csv(data_folder + 'test.csv')
        self.train = pd.read_csv(data_folder + 'train.csv')
        self.sample_submission = pd.read_csv(data_folder + 'sample_submission.csv')
#        
#data = all_data()        
#test_data = data.test
#train_data = data.train
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

one_question_pair = train_data.iloc[0]
question1 = one_question_pair['question1']
question2 = one_question_pair['question2']
tagged_question_1 = tag_sentence(question1)





#q_1_word_tokenized = word_tokenize(question1)
#q1word1 = q_1_word_tokenized[0]
#q_1_tagged = pos_tag (q_1_word_tokenized)
#q1word1 = q_1_tagged[0][1]
#simplifiedtag = [(word, map_tag('en-ptb','universal',tag)) for word, tag in q_1_tagged]
#
#doubletag = pd_DataFrame ({d})
#
#print ( simplifiedtag )
#
#q_2_word_tokenized = word_tokenize(question2)


