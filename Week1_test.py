# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:24:51 2017

@author: User-PC
"""

import pandas
import numpy as np
from scipy.stats.stats import pearsonr

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

man_count = 0
women_count = 0

sex_arr = data['Sex']

for x in sex_arr:
    if x == 'male':
        man_count += 1
    else:
        women_count += 1
print(str(man_count) + ' ' + str(women_count))

survived_arr = data['Survived']

survived_amount = 0

for x in survived_arr:
    if x == 1:
        survived_amount += 1

survived_persentage = int(survived_amount / survived_arr.size * 100)

print(survived_persentage)

first_class_arr = data['Pclass']
first_class_amount = 0

for x in first_class_arr:
    if x == 1:
        first_class_amount += 1
    
print(int(first_class_amount / first_class_arr.size * 100))


sibsp_arr = data['SibSp']
parch_arr = data['Parch']
print(pearsonr(sibsp_arr, parch_arr))
        

def ExtractName(full_name):
    first_name = ''
    if full_name.find('(') != -1:
        name_splitted = full_name.split('(')
        name_splitted1 = name_splitted[1].split(' ') 
        first_name = name_splitted1[0]
    else :
        name_splitted = full_name.split(' ')
        first_name = name_splitted[2]
    return first_name

def ClearName(name):
    if ')' in name:
        name = name.replace(')', '')
    if '.' in name:
         name = name.replace('.', '')
    if '"' in name:
         name = name.replace('"', '')    
    return name

def FindMax(names, amounts):
    max_c = 0
    max_ind = 0
    for i in range(len(names)):
        if amounts[i] > max_c:
            max_c = amounts[i]
            max_ind = i
    return names[max_ind]

name_arr = np.array(data['Name'])
name_arr1 = []
sex_arr1 = np.array(data['Sex'])
j = 0
for i in range(len(name_arr)):
    if sex_arr1[i] == 'female':
        name_arr1.append(ClearName(ExtractName(name_arr[i])))
name_counts = []
for x in name_arr1:
    n = 0
    for y in name_arr1:
        if x == y:
            n += 1
    name_counts.append(n)

print(FindMax(name_arr1, name_counts))