#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 19:17:25 2017

@author: joe
"""
import requests
import os 
import zipfile

os.listdir(os.environ['HOME'])
os.environ['HOME']

# The direct link to the Kaggle data set
data_url = 'https://www.kaggle.com/c/quora-question-pairs/download/train.csv.zip'

# The local path where the data set is saved.
localdir = os.environ['HOME'] + '/repos/qqp/' 
local_filename = localdir + 'train.zip'

# Kaggle Username and Password
kaggle_info = {'UserName': "joesadaka@gmail.com", 'Password': "24hqz510219A!"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get(data_url)

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info)

# Writes the data to a local file one chunk at a time.
with open (local_filename, 'wb') as writefile:
    for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
        if chunk: # filter out keep-alive new chunks
            writefile.write(chunk)
    writefile.close()
    
zip_ref = zipfile.ZipFile(local_filename,'r')
os.remove(local_filename)
zip_ref.extractall(localdir)