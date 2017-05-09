#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:55:50 2017

@author: kaggle
"""
import sys
import os
from queue import Queue
import threading 
#import time 
scriptpath = os.environ['HOME']  + '/repos/qqp/scripts'
data_folder = os.environ['HOME']  + '/repos/qqp/'
sys.path.append(scriptpath)
import qqp_features as qqpf
import pandas as pd
import numpy as np
import math
import time 

class threaded_results(qqpf.all_data):
    def __init__(self):
        super(threaded_results, self).__init__(False,True)
        self.processed_data = pd.DataFrame()
        self.errors = pd.DataFrame()
        self.q = Queue()
        self.print_lock = threading.Lock()
        self.append_lock = threading.Lock()
    
    def queue_chunks(self, n_chunks= np.nan):
        if math.isnan(n_chunks):
            self.n_chunks = self.max_train_chunks
        else:
            self.n_chunks = n_chunks
        for i in range(self.n_chunks):
            self.q.put(i)
        return None    
    
    def process_a_chunk(self,worker, thread_name):
#        with self.print_lock:
#            print('processing chunk {} out of {} in  || {} ||'.format(worker,self.n_chunks,thread_name))
        self.processed_chunk , self.error_chunk = self.process_training_feature_chunk_v2()        
        with self.append_lock:
            self.processed_data = self.processed_data.append(self.processed_chunk)
            self.errors = self.errors.append(self.error_chunk)
        return self
    
    def threader(self):
        while True:
            worker = self.q.get()            
            self = self.process_a_chunk(worker,threading.current_thread().name)                        
            self.q.task_done()
        return None
    
    def run_threads(self,num_threads = np.nan):
        if math.isnan(num_threads):             
            num_threads = self.max_train_chunks
        self.reset_train_chunks()
        qitems = self.q.qsize()
        start = time.time()
        for i in range(num_threads):
            t = threading.Thread(target = self.threader)
            t.name  = 'data processing thread {}'.format(i)
            t.daemon = True
            t.start()        
        self.q.join()
        self.processed_data = self.processed_data.sort_index()
        end = time.time()
        runtime = end - start
#        print('Entire job took: {} ' .format(runtime))
        return {"start_time" :start, "end_time" : end,"run_time": runtime , "num_threads":num_threads,"queue_items":qitems}