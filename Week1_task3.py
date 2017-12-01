# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:08:57 2017

@author: User-PC
"""

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

f = open('test3.txt', 'w')

first_class_arr = data['Pclass']
first_class_amount = 0

for x in first_class_arr:
    if x == 1:
        first_class_amount += 1
    
print(first_class_amount / first_class_arr.size * 100)
f.write(str(round(first_class_amount / first_class_arr.size * 100, 2)))
f.close()