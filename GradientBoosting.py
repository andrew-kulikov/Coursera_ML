import pandas
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def Sigma(pred):
    for i in range(len(pred)):
        pred[i][0] = 1.0 / (1.0 + np.e**(-pred[i][0]))
    return pred

data = pandas.read_csv('Data/gbm.csv')

y = np.array(data['Activity'])
X = np.array(data.loc[:, 'D1':'D1776'])

X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.8,
        random_state=241)

min_log_loss = 100
min_ll_iter = 0

for rate in [1, 0.5, 0.3, 0.2, 0.1] :
    clf = GradientBoostingClassifier(n_estimators=250,
                                     verbose=True,
                                     random_state=241,
                                     learning_rate=rate)
    clf.fit(X_train, y_train)
    test_score = np.zeros(250)
    train_loss = np.zeros(250)
    test_loss = np.zeros(250)
    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        train_loss[i] = log_loss(y_train, Sigma(y_pred))
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_loss[i] = log_loss(y_test, Sigma(y_pred))
        if rate == 0.2 and test_loss[i] < min_log_loss:
            min_log_loss = test_loss[i]
            min_ll_iter = i
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])

forest = RandomForestClassifier(n_estimators=min_ll_iter, random_state=241)
forest.fit(X_train, y_train)
forest_log_loss = log_loss(y_test, forest.predict_proba(X_test))


fout = open('Answers/GradMinLoss.txt', 'w')
print(min_log_loss, min_ll_iter, end='', file=fout)
fout = open('Answers/GradFiting.txt', 'w')
print('overfitting', end='', file=fout)
fout = open('Answers/GradVsForest.txt', 'w')
print(forest_log_loss, end='', file=fout)
fout.close()