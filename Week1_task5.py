# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:49:22 2017

@author: User-PC
"""

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

f = open('test5.txt', 'w')

sibsp_arr = data['SibSp']
parch_arr = data['Parch']
print(data.corr()['SibSp'][4])
f.write(str(data.corr()['SibSp'][4]))
f.close()
#print(pearsonr(sibsp_arr, parch_arr))