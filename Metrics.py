import sklearn.metrics as metrics
import pandas


def FindOptimalPrecisionRecall(true_ans, classifier_ans):
    pres, recall, thresholds = metrics.precision_recall_curve(
            true_ans,
            classifier_ans
            )
    ans = list(zip(pres, recall, thresholds))
    x = 0
    for i in range(len(ans) - 1, 0, -1):
        if ans[i][1] >= 0.7 and ans[i][0] > x:
            x = ans[i][0]
    return x
    
data = pandas.read_csv(
        'Data/classification.csv', 
        index_col=False,
        )

true = data['true']
pred = data['pred']

tp, fp, fn, tn = 0, 0, 0, 0

for i in range(len(true)):
    if pred[i] == 1:
        if true[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if true[i] == 0:
            tn += 1
        else:
            fn += 1

fout = open('Answers/Metrics1.txt', 'w')
print(tp, fp, fn, tn, end='', file=fout)
fout.close()

acc = metrics.accuracy_score(true, pred)
prec = metrics.precision_score(true, pred)
recall = metrics.recall_score(true, pred)
fScore = metrics.f1_score(true, pred)

fout = open('Answers/Metrics2.txt', 'w')
print(acc, prec, recall, fScore, end='', file=fout)
fout.close()

data = pandas.read_csv('Data/scores.csv')
true = data['true']
score_logreg = data['score_logreg']
score_svm = data['score_svm']
score_knn = data['score_knn']
score_tree = data['score_tree']

s_logreg = metrics.roc_auc_score(true, score_logreg)
s_svm = metrics.roc_auc_score(true, score_svm)
s_knn = metrics.roc_auc_score(true, score_knn)
s_tree = metrics.roc_auc_score(true, score_tree)
names = ['score_logreg',
         'score_svm',
         'score_knn',
         'score_tree']
roc = sorted(list(zip(
         names,
         [s_logreg,
          s_svm,
          s_knn,
          s_tree])),
        key = lambda t: t[1])

fout = open('Answers/Metrics3.txt', 'w')
print(roc[-1][0], end='', file=fout)
fout.close()

x_logreg = FindOptimalPrecisionRecall(true, score_logreg)
x_svm = FindOptimalPrecisionRecall(true, score_svm)
x_knn = FindOptimalPrecisionRecall(true, score_knn)
x_tree = FindOptimalPrecisionRecall(true, score_tree)
fout = open('Answers/MetricsOptimalScore.txt', 'w')
x = sorted(
        list(
                zip(
                        names,
                        [x_logreg, x_svm, x_knn, x_tree]
                        )
                ),
                key=lambda t:t[1]
            )
print(x[-1][0], end='', file=fout)
fout.close() 