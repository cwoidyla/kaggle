# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:10:59 2017

@author: Conrad
"""
#%% Import libraries
import os
import sys
import csv
import numpy
import pandas as pd
import time
#%% Change to data directory
# TODO: Make this a function. Pass in a directory path
print(os.getcwd())      # view current working directory
path = "C:\\Users\\Owner\\Desktop\\MGMT552\\Santander"
os.chdir(path)          # change working directory
print(os.getcwd())      # view current working directory
#%% Load data into columns
# TODO: Make this a function. Pass in file name.
def load_to_data_columns(file):
  print(os.listdir())            # view files in directory
  f = open(file, 'r')
  reader = csv.reader(f)
  headers = next(reader,None)
  column = {}
  for h in headers:
    column[h] = []
  for row in reader:
    for h, v in zip(headers, row):
      column[h].append(v)
  f.close()
  return column

test_data = load_to_data_columns('test_ver2.csv')
train_data = load_to_data_columns('train_ver2.csv')
#%% Load data into numpy data type
def load_to_numpy_matrix(file):
  beg_time = time.time()
  arr = numpy.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)
  end_time = time.time()
  print(end_time - beg_time)
  return arr

temp_data = load_to_numpy_matrix('test_ver2.csv')

#%% Load data using pandas
def load_with_pandas(file):
  beg_time = time.time()
  df = pd.read_csv(file)
  #saved_column = df.column_name #you can also use df['column_name']
  end_time = time.time()
  print(end_time - beg_time)
  return df

test_data = load_with_pandas("test_ver2.csv")
train_data = load_with_pandas("train_ver2.csv")


