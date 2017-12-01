import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def GetAccurancy(X, y):
    maxAccurncy = 0.0
    goodK = 0
    for k in range(1, 51):
        neigh = KNeighborsClassifier(n_neighbors=k)
        amountOfRightPredictions = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            neigh.fit(X_train, y_train)
            pred = neigh.predict(X_test)    
            for i in range(len(pred)):
                if pred[i] == y_test[i]:
                    amountOfRightPredictions += 1
        if amountOfRightPredictions / len(y) > maxAccurncy:
            maxAccurncy = amountOfRightPredictions / len(y)
            goodK = k
            #print('Test #', i, '.\nPrediction: ', *pred, '\nReal: ', *y_test, sep='')
            #i += 1
            #plt.plot(X_test, pred, 'ro', X_test, y_test, 'g^')
            #plt.show()
    return maxAccurncy, goodK

data = pandas.read_csv('Data/Wine.csv', index_col=False)
X = []
y = np.array(data['0'])
for i in range(len(data)):
    X.append([])
    for j in range(1, len(data.count())):
        X[i].append(data[str(j)][i])

kf = KFold(5, shuffle=True, random_state=42)
X = np.array(X)

fout = open('KNN.txt', 'w')

maxAccurncy, goodK = GetAccurancy(X, y)
print(maxAccurncy, goodK, file=fout, end=' ')

XScaled = scale(X)

maxAccurncy, goodK = GetAccurancy(XScaled, y)
print(maxAccurncy, goodK, file=fout, end=' ')

fout.close()