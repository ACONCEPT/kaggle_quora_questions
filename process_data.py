#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:46:56 2017

@author: joe
"""
import time
import qqp_features as qqp
from qqp_features import tag_sentence, count_matching_noun_verb, count_matching_noun_verb_lemma
filename  = 'train.csv' #404290 rows
data_folder = os.environ['HOME']  + '/repos/qqp/'

class all_data(object):
    """
all_data, bring over the feature processing from qqp_features, in this first 
loop where we enumerate from the listform of the imported data, process the data
immediately into the features there, load them into a list of dictionaries. 
place each feature into a list, do not allow the data to load on initialization of 
the object anymore, instead, put into the init the control flow process for 
making the df. once all the processing is done, write the features to csv.    
    """
    def __init__(self,num_rows,filename, test = False, testsize = 500000, test_loops = 1000):        
        self.num_rows = num_rows
        self.filename = filename
        self.test = test
        self.testsize = testsize
        self.test_loops = test_loops

    def pull_data(self):
        if self.test: 
            print ('data test mode, pulling {} rows '.format(testsize))
        else:
            print ('loading full log.txt file... ')
        with open(filename,'r',encoding = 'utf-8',errors = 'ignore') as data_file:
            bigdata = data_file.read() 
        listform = bigdata.splitlines()
        self.df = pd.DataFrame()
        self.data = {}
        self.errors = {}
        s = time.time()
        self.columns =  listform.pop(0)
        for i, row in enumerate(listform): 
            try:
                print(type(row))
                print(row)
                if i == 10:
                    break
            except Exception as e:
                myexception = {}                 
                myexception['Exception'] = e
                myexception['row']= row
                myexception ['failure_time'] = time.time() - s
                self.errors[i] = myexception                                   
        self.load_time = time.time()- s 
        self.rows = i 
        print ('total rows : {}'.format(i))        
        s = time.time()
        return self
        
data = all_data(10, data_folder + filename, False)
data.pull_data()
print (data.num_rows,data.filename, data.test, data.testsize,data.test_loops, data.columns)




                
#actually makes the dataframe
    def make_df(self):                
        print("making DataFrame...")
        s = time.time()
        self.df = pd.DataFrame(self.for_df,index = self.indx)
#        self.df = pd.DataFrame(self.for_df)
        self.make_df_time = time.time() - s
        return None
    
# turns the dictionary of parsed logs into a series lof lists that can be used to make a dataframe
    def make_lists(self):
        print("making lists...")
        self.host = []
        self.time_received = []
        self.method = []
        self.url = []
        self.http_ver = []
        self.response = []
        self.bytes = []
        self.logs = [] 
        self.unique_hosts = [] 
        self.unique_resources = []         
        self.unique_responses = [] 
        self.unique_methods = [] 
        s = time.time()
        for i, di in self.parsed_data.items():
            self.host.append(di['host'])           
            self.time_received.append(split_date(di['time_received']))
            self.method.append(di['method'])
            self.url.append(di['url'])
            self.http_ver.append(di['http_ver'])
            self.response.append(di['response'])
            self.bytes.append(di['bytes'])
            self.logs.append(di['log'])
        check = len(self.host) == len(self.time_received) == len(self.method) == len(self.url) == len(self.http_ver) == len(self.response) == len(self.bytes)
        print (' lists check : {}'.format(check))
        self.cols.remove('time_received')
        self.for_df = dict(zip(self.cols,[self.host,self.method,self.url,self.http_ver,self.response, self.bytes,self.logs]))        
        self.indx = pd.DatetimeIndex(self.time_received)
        end = time.time()
        self.make_lists_time =  end-s        
        return None
    
# any items not processed by the fast parser are kept separate, verified, and rejoined
    def integrate_errors(self):
        print("integrating errors...")
        self.host = []
        self.time_received = []
        self.method = []
        self.url = []
        self.http_ver = []
        self.response = []
        self.bytes = []
        self.logs = []         
        s = time.time()
        for i, di in self.diff_rows.items():
            try:
                self.host.append(di['remote_host'].rstrip('- -'))               
            except NameError:
                self.host.append(np.nan)
            try:
                self.time_received.append(split_date(di['time_received_tz_isoformat']))
            except NameError:
                self.time_received.append(np.nan)
            try:
                self.method.append(di['request_method'])
            except NameError:
                self.method.append(np.nan)
            try:            
                self.url.append(di['request_url'])
            except NameError:
                self.url.append(np.nan)
            try:
                self.http_ver.append(di['request_http_ver'])
            except NameError:
                self.http_ver.append(np.nan)
            try:
                self.response.append(int(di['status']))
            except NameError:
                self.response.append(np.nan)
            try:
                self.bytes.append(int(di['response_bytes_clf']))
            except NameError:
                self.bytes.append(np.nan)
            except ValueError:
                self.bytes.append(0)
            try:
                self.logs.append(di['log'])
            except NameError:
                self.logs.append(np.nan)
        check = len(self.host) == len(self.time_received) == len(self.method) == len(self.url) == len(self.http_ver) == len(self.response) == len(self.bytes) == len(self.logs)
        print (' lists check : {}'.format(check))
#        self.cols.remove('time_received')
        self.for_df = dict(zip(self.cols,[self.host,self.method,self.url,self.http_ver,self.response, self.bytes,self.logs]))        
        self.indx = pd.DatetimeIndex(self.time_received)
        self.irrdf = pd.DataFrame(self.for_df,index = self.indx)
        self.df = self.df.append(self.irrdf)
        end = time.time()
        self.make_lists_time =  end-s        
        return None
    
#    control data flow and processing of features
    def make_features(self,output_folder, f1 =False, f2 = False, f3 =False,f4 = False):
        self.output_folder = output_folder
        if f1:
            self.feature1(output_folder)
        if f2:
            self.feature2(output_folder)        
        self.df.sort_index()
        self.df = self.df.reset_index()
        self.df = self.df.rename(index = str, columns= {'index':'time_received'})    
        if f3:
            self.feature3(output_folder)
        if f4:
            self.feature4(output_folder) 
            
            
    #hotss visiting site most often
    def feature1(self,output_folder):
        print('making feature 1...' )
        s = time.time()
        self.f1 = top_active_host(self.df)
        e = time.time()
        self.f1_time = e - s
        self.f1.to_csv(output_folder + 'hosts.txt',header = False)
        return None
            
    # urls using most bandwidth
    def feature2(self, output_folder):
        print('making feature 2...' )
        s = time.time()
        self.f2, self.f2_errors = top_bandwidth_resource(self.df)
        e = time.time() 
        self.f2_time = e - s 
        self.f2.to_csv(output_folder + 'resources.txt', header = False)
        return None
    
    # most number of visits in 60 minutes
    def feature3(self,output_folder):
        print('making feature 3...' )
        s = time.time()
        self.f3, self.f3_errors = busiest_timeframe(self)
        e = time.time()
        self.f3_time = e-s 
        self.f3.to_csv(output_folder + 'hours.txt',header = False)
        return None
        
    
    # if 3 failed attempts in 20 seconds, block for 5 minutes 
    def feature4(self,output_folder):
        print('making feature 4...' )
        s = time.time()
        self.f4 , self.f4_errors  = blocked(self.df,self.time_received)
        e = time.time() 
        self.f4_time = e-s
        with open(output_folder + 'blocked.txt', 'w') as outfile:
            for item in self.f4:
                outfile.write(item + '/n')
#        self.f4.to_csv(output_folder + 'blocked.txt',header = False)
        return None
