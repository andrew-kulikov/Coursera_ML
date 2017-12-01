import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

fout = open('DecisionTreeAnswer.txt', 'w')

newData = []
y = []

Pclass = list(data['Pclass'])
Fare = list(data['Fare'])
Age = list(data['Age'])
Sex = list(data['Sex'])
Y = list(data['Survived'])

for i in range(len(data)):
    nowSex = 1
    if Sex[i] == 'male':
        nowSex = 0
    nowPass = [Pclass[i], Fare[i], Age[i], nowSex]
    good = True
    for x in nowPass:
        if np.isnan(x):
            good = False
    if good:
        newData.append(nowPass)
        y.append(Y[i])



clf = DecisionTreeClassifier(random_state=241)
clf.fit(newData, y)


x = zip(['Pclass', 'Fare', 'Age', 'Sex'], clf.feature_importances_)

print(*x)

fout.close()
