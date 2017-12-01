import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
        )

f = open('Data.txt', 'w', encoding='utf-8')
print(newsgroups, file=f)
f.close()

vectorizer = TfidfVectorizer()
doc = vectorizer.fit_transform(newsgroups.data)

feature_mapping = vectorizer.get_feature_names()

cFile = open('OptimalCForTextAnalyzer.txt', 'r')
C = int(cFile.read())

if C == -1:
    grid = {'C' : np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(doc, newsgroups.target)
    
    m = 0
    for a in gs.grid_scores_:
        x = a.mean_validation_score
        if x > m:
            m = x
            C = a.parameters['C']
    cFile = open('OptimalCForTextAnalyzer.txt', 'w')
    print(C, file=cFile)
    cFile.close()

clf = SVC(C=C, kernel='linear', random_state=241)
clf.fit(doc, newsgroups.target)

fout = open('Answers/Texts.txt', 'w', encoding='utf-8')
x = list(zip(np.absolute(clf.coef_.toarray()[0]), feature_mapping))
words = []
for a in sorted(x, key=lambda t: t[0])[-10:]:
    words.append(a[1])
words.sort()
print(*words, end='', file=fout)
fout.close()
