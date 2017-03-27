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

results = threaded_results()
results.change_chunk_size(100)
results.queue_chunks(50)
print('chunk_size = {}, max_chunks = {}'.format(results.chunk_size,results.max_train_chunks))
print("qsize : {}, qempty : {}, qfull : {}".format(results.q.qsize(),results.q.empty(),results.q.full()))
#runinfo = results.run_threads()

def case_1():
    results = threaded_results()
    results.change_chunk_size(10)
    results.queue_chunks(10)
    runinfo = results.run_threads(2)
    return runinfo

def case_2():
    results = threaded_results()
    results.change_chunk_size(10)
    results.queue_chunks(10)
    runinfo = results.run_threads(4)
    return runinfo

def case_3():
    results = threaded_results()
    results.change_chunk_size(10)
    results.queue_chunks(10)
    runinfo = results.run_threads(8)
    return runinfo

def case_4():
    results = threaded_results()
    results.change_chunk_size(10)
    results.queue_chunks(10)
    runinfo = results.run_threads(8)
    return runinfo

def case_5():
    results = threaded_results()
    results.change_chunk_size(10)
    results.queue_chunks(10)
    runinfo = results.run_threads()
    return runinfo


case1_runinfo = pd.DataFrame()
case2_runinfo = pd.DataFrame()
case3_runinfo = pd.DataFrame()
case4_runinfo = pd.DataFrame()
case5_runinfo = pd.DataFrame()

runinfo = case_1()

for i in range(1000):
    print (' case test {} out of {} '.format(i,1000))
    runinfo = case_1()
    runinfo2 = case_2()
    runinfo3 = case_3()
    runinfo4 = case_4()
    runinfo5 = case_5() 
    case1_runinfo = case1_runinfo.append(runinfo,ignore_index = True)
    case2_runinfo = case2_runinfo.append(runinfo2,ignore_index = True)
    case3_runinfo = case3_runinfo.append(runinfo3,ignore_index = True)
    case4_runinfo = case4_runinfo.append(runinfo4,ignore_index = True)
    case5_runinfo = case5_runinfo.append(runinfo5,ignore_index = True)
    