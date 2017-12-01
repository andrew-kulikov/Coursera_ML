import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score


def Proba(x, w):
    return 1 / (1 + math.exp(-w[0] * x[0] - w[1] * x[1]))

def GetSum(X, y, w, num):
    s = 0
    for j in range(len(X)):
        s += y[j] * X[j][num] * (1 - 1 / (1 + math.exp(-y[j] * (w[0] * X[j][0] + w[1] * X[j][1]))))
    return s

def Grad(X, y, w=[0, 0], k=0.1, C=1, max_steps=10000, e=10**-5):
    l = len(X)
    i = 0
    while i < max_steps:
        w0 = w[0] + k * (1 / l) * GetSum(X, y, w, 0) - k * C * w[0]
        w1 = w[1] + k * (1 / l) * GetSum(X, y, w, 1) - k * C * w[1]
        w_old = [0, 0]
        w_old[0] = w[0]
        w_old[1] = w[1]
        w[0] = w0
        w[1] = w1
        if distance.euclidean(w_old, w) <= e:
            break
    return w

def GetScores(X, w):
    scores = []
    for x in X:
        scores.append(Proba(x, w))
    return scores

data = pandas.read_csv(
        'Data/data-logistic.csv',
        index_col=False,
        header=None
        )

y = np.array(data[0])
X = np.array(data.loc[:, 1:])

x1 = data[1]
x2 = data[2]

for i in range(len(y)):
    s = 'ro'
    if y[i] == 1:
        s = 'go'
    plt.plot(x1[i], x2[i], s)

fout = open('Answers/Logistic.txt', 'w')

w = Grad(X, y, C=0)
y_scores = GetScores(X, w)
print(roc_auc_score(y, y_scores), file=fout, end=' ')

print(w, y_scores[:5])

w = Grad(X, y, C=10)
y_scores = GetScores(X, w)
print(roc_auc_score(y, y_scores), file=fout, end=' ')

print(w, y_scores[:5])

fout.close()
plt.show()