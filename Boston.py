import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

boston = load_boston()

X = scale(boston.data)
y = boston.target

kf = KFold(n_splits=5, shuffle=True, random_state=42)

maxScore = -100.0
optimalP = 0

for i in np.linspace(1, 10, 200):
    reg = KNeighborsRegressor(n_neighbors=5,
                              weights='distance',
                              p=i)
    scores = cross_val_score(reg,
                          X,
                          y,
                          scoring='neg_mean_squared_error',
                          cv=kf
                          )
    print(scores.mean())
    if scores.mean() > maxScore:
        maxScore = scores.mean()
        optimalP = i



fout = open('Answers/Boston.txt', 'w')
print(maxScore, optimalP, file=fout, end='')
fout.close()
