import pandas
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

train = pandas.read_csv('Data/salary-train.csv')
test = pandas.read_csv('Data/salary-test-mini.csv')
y_train = train['SalaryNormalized'] 
y_test = test['SalaryNormalized']

enc = DictVectorizer()
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
features = train[['LocationNormalized', 'ContractTime']].to_dict('records')
X_train_categ = enc.fit_transform(features).toarray()

train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

descriptions = np.array(train['FullDescription'])

for i in range(len(descriptions)):
    descriptions[i] = descriptions[i].lower()

tv = TfidfVectorizer(min_df=5)
doc = tv.fit_transform(descriptions)

X_train = hstack((doc, X_train_categ))

ridge = Ridge(random_state=241)
ridge.fit(X_train, y_train)

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)
features = test[['LocationNormalized', 'ContractTime']].to_dict('records')
X_test_categ = enc.transform(features).toarray()
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

descriptions = np.array(test['FullDescription'])

for i in range(len(descriptions)):
    descriptions[i] = descriptions[i].lower()

doc = tv.transform(descriptions)

X_test = hstack((doc, X_test_categ))

fout = open('Answers/SalaryPrediction.txt', 'w')
pred = ridge.predict(X_test)
print(*pred, file=fout, end='')
fout.close()