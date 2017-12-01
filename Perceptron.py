import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pandas.read_csv(
        'Data/perceptron-train.csv',
        index_col=False,
        header=None)

test_data = pandas.read_csv(
        'Data/perceptron-test.csv',
        index_col=False,
        header=None)

scaler = StandardScaler()

X_train = np.array(train_data.loc[:, 1:])
y_train = np.array(train_data[0])
X_train_scaled = scaler.fit_transform(X_train)

X_test = np.array(test_data.loc[:, 1:])
y_test = np.array(test_data[0])
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

clf.fit(X_train_scaled, y_train)
pred_on_scaled = clf.predict(X_test_scaled)

acc1 = accuracy_score(y_test, pred)
acc2 = accuracy_score(y_test, pred_on_scaled)

fout = open('Answers/Perceptron.txt', 'w')
print(acc2 - acc1, end='', file=fout)
fout.close()
