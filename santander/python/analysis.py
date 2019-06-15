# -*- coding: utf-8 -*-
"""
Created on Sat May  6 12:38:34 2017

@author: Conrad
"""
loop = False
import dataset
test = dataset.SantanderAnalysis("renta_samples/2015-12-28.csv")
mine = dataset.MineData(test.pred_vars)
results = []
if (loop):
    #if (cluster == "prototype")
    for i in range(4,10):
        test.clusters = mine.k_prototype(i, test.clustees)
        results.append(dataset.np.unique(test.clusters, return_counts = True))
    for result in results:
        print(result)
else:
    #test.clusters = mine.k_prototype(8, test.clustees)
    test.clusters = mine.k_modes(8)