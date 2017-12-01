# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:06:36 2017

@author: User-PC
"""

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

f = open('test1.txt', 'w')

man_count = 0
women_count = 0

sex_arr = data['Sex']

for x in sex_arr:
    if x == 'male':
        man_count += 1
    else:
        women_count += 1
print(str(man_count) + ' ' + str(women_count))
f.write(str(man_count) + ' ' + str(women_count))
f.close()