# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:10:52 2017

@author: User-PC
"""

import pandas
import math
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

f = open('test4.txt', 'w')

all_ages = data['Age']
ages = []
for x in all_ages:
    if math.isnan(x) != True:
        ages.append(x)
print(str(round(np.mean(ages), 2)) + ' ' + str(np.median(ages))) 
f.write(str(round(np.mean(ages), 2)) + ' ' + str(np.median(ages)))
f.close()