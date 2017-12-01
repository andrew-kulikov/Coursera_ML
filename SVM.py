import numpy as np
import pandas
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = pandas.read_csv('data/svm-data.csv',
                       index_col=False,
                       header=None)

clf = SVC(C=100000, random_state=241, kernel='linear')

X = np.array(data.loc[:, 1:])
y = np.array(data[0])

p1 = data[1]
p2 = data[2]
for i in range(len(y)):
    color = 'go'
    if y[i] == 1:
        color = 'ro'
    plt.plot(p2[i], p1[i], color)


clf.fit(X, y)
plt.show()

fout = open('Answers/SVM.txt', 'w')
print(*clf.support_, sep=',', end='', file=fout)
fout.close()