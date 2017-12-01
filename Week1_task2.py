# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:07:54 2017

@author: User-PC
"""

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

f = open('test2.txt', 'w')

survived_arr = data['Survived']

survived_amount = 0

for x in survived_arr:
    if x == 1:
        survived_amount += 1

survived_persentage = round(survived_amount / survived_arr.size * 100, 2)

#print(survived_persentage)
f.write(str(survived_persentage))
f.close()