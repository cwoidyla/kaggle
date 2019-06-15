# -*- coding: utf-8 -*-
"""
Created on Mon May  1 06:53:38 2017

@author: Conrad
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import multiprocessing
from kmodes import kprototypes
from kmodes import kmodes

#%%
class Dataset(object):
    def __init__(self):
        self.data = pd.Series()
        
    def load_data(self, file):
        print("Loading data...")
        df = pd.read_csv(file, encoding="latin-1")
        print("Loading complete")
        return df
    
    def write_data(self, data, filename):
        print("In " + os.getcwd())
        print("Writing to " + filename)
        data.to_csv(filename, index=False)
        print("Finished writing " + filename)
    
    def sample_data(self, fraction):
        sample_set = self.data.sample(frac = fraction)
        print("Sample data set created")
        return sample_set
    
    def write_split_data(self, data, column):
        # Data is split w.r.t. unique vals in column.
        unique_vals = data[column].unique()
        for value in unique_vals:
            filename = value + ".csv"
            print("writing " + filename)
            self.write_data(data[data[column].str.contains(value)],
                            filename)
    
#%%
class CleanData(object):
    # Looks at most often occuring value and prints its ratio
    def suggest_replacement_value(data, column, suggestions):
        print("NaNs: " + str(data[column].isnull().sum() ))
        print("NaN Ratio: " 
              + str(data[column].isnull().sum()/len(data[column] ) ) )
        temp_counts = data[column].value_counts()
        print("Unique values: " + str(len(temp_counts)) )
        print(temp_counts.head(suggestions))
        
    def clean_nans(data, column, value):
        data[column] = data[column].replace(np.NaN, value)
    
    def replace_col_value(data, column, old_val, new_val):
        print("Replacing column " + column + " values...")
        data[column].loc[data[column] == old_val] = new_val
    
    def format_col_dtype(data, column, data_type):
        # not all data in column is of same type.
        # converts to string, strips whitespace, converts NA to np.nan
        # converts fixed column to float
        print("Formatting column " + column + " data type..")
        data[column] = data[column].astype(str)
        data[column] = CleanData.remove_whitespace(data[column])
        data[column][data[column].str.contains('NA')] = np.nan
        #self.pred_var["indrel_1mes"].loc[pred_var.indrel_1mes == "P"] = 5
        if(data_type == "float"):
            data[column] = data[column].astype(float)
        elif(data_type == "int"):
            data[column] = data[column].astype(int)
    
    def remove_whitespace(data):
        return data.str.strip()
    
    def remove_nan_rows(data, column):
        # remove row based on column with nan value
        return data[data[column].notnull()]
    
    def nan_replace_impact(data, nan_col, affected_col):
        # gets ratio of nans replaced w.r.t. another column
        mod_data = data[affected_col][data[nan_col].notnull()]
        orig_counts = data[affected_col].value_counts()
        mod_counts = mod_data.value_counts()
        return (orig_counts - mod_counts) / orig_counts

#%%
class MineData(object):
    def __init__(self, data):
        self.data = data
        #self.data = data.as_matrix()
    
    def convert_col_type(self, data, indices):
        cols = data[indices].columns # get category cols
        for c in cols:
            if data[c].dtype != 'O':
                print("Converting " + c + " to string")
                data[c] = data[c].astype(str)
            else:
                print("Skipping " + c)
        return data
    
    def get_cat_cols(self, data, num_cols):
        cat_data_indices = list(range(len(data.columns)))
        #cat_data_indices = list(range(self.data.shape[1])) # get column num
        for col in num_cols:
            cat_data_indices.remove(col) #remove from list
        return cat_data_indices
    
    def k_prototype(self, clust_num, clustees):
        print("Starting k-prototypes clustering...")
        kproto = kprototypes.KPrototypes(n_clusters = clust_num, 
                                         init='Cao', verbose = 2)
        num_cols = [4,21] # age, renta
        cat_data_indices = self.get_cat_cols(self.data, num_cols)
        self.data = self.convert_col_type(self.data, cat_data_indices)
        #print(self.data.dtypes)
        clusters = kproto.fit_predict(self.data.values, 
                                      categorical=cat_data_indices)
        print ("cluster centroids of the trained model.")
        print (kproto.cluster_centroids_)       
        print ("training statistics")
        print (kproto.cost_)
        print (kproto.n_iter_)
        
        #for s, c in zip(clustees, clusters):
        #    print("CustID: {}, cluster:{}".format(s, c))
        return clusters
    
    def k_modes(self, clust_num):
        kmodes_cao = kmodes.KModes(n_clusters=clust_num, init='Cao', verbose=1)
        num_cols = [4,21] # age, renta
        cat_data_indices = self.get_cat_cols(self.data, num_cols)
        self.data = self.convert_col_type(self.data, cat_data_indices)
        categorical_data = self.data[cat_data_indices] # get category cols
        print(categorical_data.dtypes)
        kmodes_cao.fit(categorical_data)
        # Print cluster centroids of the trained model.
        print('k-modes (Cao) centroids:')
        print(kmodes_cao.cluster_centroids_)
        # Print training statistics
        print('Final training cost: {}'.format(kmodes_cao.cost_))
        print('Training iterations: {}'.format(kmodes_cao.n_iter_))
        return kmodes_cao.labels_
#%%
class SantanderAnalysis(object):
    def __init__(self, filepath):
        self.pred_vars = pd.Series()
        self.clustees = pd.Series()
        self.clusters = []
        self.train_data = Dataset()
        self.train_data.data = self.train_data.load_data(filepath)
        #self.train_data.sample_data(0.0001)
        # self.fix_data()
        self.get_pred_vars(self.train_data.data, 24, 1)
        #self.clusters = MineData.k_prototype(self.pred_vars, 
        #                                     8, 
        #                                     self.clustees)
    
    def fix_data(self, data, column, old_dtype, new_dtype):
        #5, 8, 11, 15 have mixed types
        #age, antiguedad, indrel_1mes, conyuemp
        CleanData.format_col_dtype(CleanData, self.train_data, "age", "float")
        CleanData.format_col_dtype(CleanData, self.train_data, 
                                   "antiguedad", "float")
        CleanData.replace_col_value(CleanData, self.train_data, 
                                   "indrel_1mes", "P", "5")
        CleanData.format_col_dtype(CleanData, self.train_data, 
                                   "indrel_1mes", "float")
        CleanData.format_col_dtype(CleanData, self.train_data, 
                                   "conyuemp", "float")
    def separate_by_months(data):
        Dataset.write_split_data(Dataset, data, "fecha_dato")
    
    def subset_month_data(data, directory):
        # C:\\Users\Owner\Desktop\MGMT552\Santander\Months
        os.chdir(directory)
        sample = Dataset(data)
        sample_set = sample.sample_data(0.01)
        Dataset.write_split_data(Dataset, sample_set, "fecha_dato")
    
    def get_pred_vars(self, data, total_cols, clustee_col):
        print("Getting predictor data...")
        pred_cols = list(range(total_cols)) #predictor variable columns
        pred_cols.remove(clustee_col) # remove data ids
        self.pred_vars = data[pred_cols].copy()
        clustee = [clustee_col] #pandas copy function requires list
        self.clustees = data[clustee].values.tolist()
    
    def analyze_data(self, data, clust_num, clustees):
        self.clusters = MineData.k_prototype(data, clust, clustees)
        

#%%
class Worker(multiprocessing.Process):
    def run(self, file):
        sa = SantanderAnalysis(file)
        
        

#%% Prototyping   
#data_directory = "C:\\Users\\Owner\\Desktop\\MGMT552\\Santander"
#train_data_path = data_directory + "\\train_ver3.csv"
#%% Production
#data_directory = "./"
#train_data_path = data_directory + "train_ver3.csv"
#%%
#test = Dataset(train_data_path)
# strip whitespace

#%% Load and prepare data
#data_directory = "../"
#train_data_path = data_directory + "train_ver2.csv"
#train_data = Dataset(train_data_path)
#train_data.sample_data(0.0001)
#print("Isolating predictor variables..")
#pred_var = train_data.sample_set[list(range(24))].copy()
#print("Formatting data types...")
#pred_var["indrel_1mes"].loc[pred_var.indrel_1mes == 'P'] = 5
#pred_var["indrel_1mes"] = pred_var["indrel_1mes"].astype(float)

#%% parallelize
#for i in pred_var.fecha_dato.unique():
#    print (i + " " + str(len(pred_var.loc[pred_var["fecha_dato"] == i])) )


#%% kprototypes
#print("Starting k-prototypes clustering...")
#kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
#categorical_indices = list(range(len(pred_var.columns)))
#categorical_indices.remove(5) # age
#categorical_indices.remove(22) # renta
#clusters = kproto.fit_predict(pred_var.values, 
#                              categorical=[categorical_indices])

# Print cluster centroids of the trained model.
#print(kproto.cluster_centroids_)

# Print training statistics
#print(kproto.cost_)
#print(kproto.n_iter_)




