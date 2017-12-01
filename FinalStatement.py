import pandas
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def main():
    start = datetime.datetime.now()
    
    data = pandas.read_csv('Data/dota.csv')
    
    y = data['radiant_win']
    X = data.drop(['radiant_win',
                   'duration',
                   'tower_status_radiant',
                   'tower_status_dire',
                   'barracks_status_radiant',
                   'barracks_status_dire'], axis=1)
    
    n_estimators = 30
    """
    Точности при количесве деревьев 10, 20, 30, 40, 50, 100
    0.663568394879
    0.682672314973
    0.689832076468
    0.694402695613
    0.697155505421
    0.706305863469
    Average time:  0:16:24.743392
    """
    
    X = X.fillna(0)
    X = np.array(X)
    y = np.array(y)
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True)
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    print(np.mean(cross_val_score(clf, X_scaled, y, cv=kf, scoring='roc_auc')))
    
    X_test = np.array(pandas.read_csv('Data/features_test.csv'))
    pred = clf.predict(X_test)
    print(pred[:5])
    
    print('Average time: ', datetime.datetime.now() - start)
    
if __name__ == '__main__':
    main()
