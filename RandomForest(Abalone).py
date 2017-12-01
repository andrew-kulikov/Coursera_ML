import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

data = pandas.read_csv('Data/abalone.csv')
y = np.array(data['Rings'])

X = data.loc[:, 'Sex':'ShellWeight']

X['Sex'] = X['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

min_number_of_trees = 50
X = np.array(X)

for n_tree in range(1, 51):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    forest = RandomForestRegressor(n_estimators = n_tree, random_state=1)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        forest.fit(X_train, y_train)
        pred = forest.predict(X_test)
        score = r2_score(y_test, pred)
        scores.append(score)
    scores = np.array(scores)
    if np.mean(scores) > 0.52 and n_tree < min_number_of_trees:
        min_number_of_trees = n_tree
    print(np.mean(score))

fout = open('Answers/RandomForest(Abalone).txt', 'w')
print(min_number_of_trees, end='', file=fout)
fout.close()
