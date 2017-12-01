import pandas
import numpy as np
from sklearn.decomposition import PCA

data = pandas.read_csv('Data/close_prices.csv')
data_djia = pandas.read_csv('Data/djia_index.csv')

X = data.loc[:, 'AXP':'XOM']
djia = data_djia['^DJI']

pca = PCA(n_components=10)
pca.fit(X)

disp = 0
n = 0
for i in range(len(pca.explained_variance_ratio_)):
    disp += pca.explained_variance_ratio_[i]
    n += 1
    if disp > 0.9:
        break

fout = open('Answers/PCA_NumberOfComponents.txt', 'w')
print(n, end='', file = fout)
fout.close()

X = pca.transform(X)
fc = []
for x in X:
    fc.append(x[0])
fc = np.array(fc)

fout = open('Answers/PCA_PearsonCorrelation.txt', 'w')
print(np.corrcoef(fc, djia)[0][1], end='', file=fout)
fout.close()

names = np.array(list(data.columns.values[1:]))
comp_coef = sorted(list(zip(pca.components_[0], names)))
fout = open('Answers/PCA_GoodCompany.txt', 'w')
print(comp_coef[-1][1], end='', file=fout)
fout.close()
print(*comp_coef, sep='\n')