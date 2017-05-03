#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:30:57 2017

@author: kaggle
"""
import sys
import os
import pandas as pd
scriptpath = os.environ['HOME']  + '/repos/qqp/scripts'
data_folder = os.environ['HOME']  + '/repos/qqp/'
sys.path.append(scriptpath)
from threading_v2 import threaded_results 

#results = threaded_results()
#results.change_chunk_size(100)
#results.queue_chunks(50)
#print('chunk_size = {}, max_chunks = {}'.format(results.chunk_size,results.max_train_chunks))
#print("qsize : {}, qempty : {}, qfull : {}".format(results.q.qsize(),results.q.empty(),results.q.full()))
#runinfo = results.run_threads()

def process(chunk_size,threads = 2, chunks = 10):
    # start up the object that processes the data
    results = threaded_results()
    
    # records per chunk, 
    # default 20,000 records
    if not chunk_size:
        results.change_chunk_size(100000)
    else:
        results.change_chunk_size(chunk_size)
    
    # indicates only number of chunks to be processed
    # defaults to running all chunks 
    if not chunks:
        results.queue_chunks()
    else:
        results.queue_chunks(chunks)    
        
    # argument is number of worker threads to use
    # default 1 thread per chunk.
    # second result is runinfo dictionary with running statistics
    if not threads:
        results , runinfo = results.run_threads()
    else:
        results , runinfo = results.run_threads(threads)
    
    # recording the number of threads used 
    runinfo ['threads'] = threads
    return results.processed_data, runinfo

def test_run(test_cases):
    runinfo = pd.DataFrame()
    for i, item in test_cases.items():
        chunksize, threads, chunks = item
        pdata , rundata = process(chunksize,threads,chunks)
        runinfo = runinfo.append(rundata,ignore_index = True)
    return pdata, runinfo


results, runinfo = test_run({1:(False,False,False)})

#runinfo = pd.DataFrame()
#for i in range(1000):
#    print (' case test {} out of {} '.format(i,1000))
#    test_cases = {1 : (10,10,2), 2:(10,10,4), 3:(10,10,6),4:(10,10,8),5:(10,10,10)}#,6:(False,False,100),7:(False,False,500)}    
#    rundata = test_run(test_cases)
#    runinfo = runinfo.append(rundata,ignore_index = True)    
#runinfo.to_csv(data_folder)
