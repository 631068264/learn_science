#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/8 15:16
@annotation = ''
"""
from sklearn.svm import SVC

"""
when classes are imbalanced, accuracy is not a great evaluation measure.
"""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

digits = load_digits()
if False:
    y = digits.target == 9
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, y, random_state=0)
    logreg = LogisticRegression(C=0.1)
    logreg.fit(X_train, y_train)
    pred_logreg = logreg.predict(X_test)
    print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))
"""
TF -->预测结果方向与真实结果方向 相同为T
PN 定义正负类别
PN--> 预测结果的类别

TN negative true 
    真假阴阳性
    TN FP
    FN TP
    
预测正确的结果所占的比例
Accuracy = (TP+TN) /(TP+TN+FP+FN)

精准率 = 被正确识别为正类别次数/(预测为正例)
Precision = TP /(TP+FP)

召回率 = 被正确识别为正列次数/(真实正例)
Recall = TP /(TP+FN)

分类阈值的调整 使得精确率和召回率往往是此消彼长
分类阈值    FP      FN
+          -       +
-          +       -

while precision and recall are very important measures
    f-score = 2PR /(R+P)

If we only looked at the f1-score to compare overall performance, we would have missed these subtleties. 
The f1-score only captures one point on the precision-recall curve, the one given by the default threshold:

缺点:
A disadvantage of the f-score, however, is that it is harder to interpret and explain than accuracy
"""
"""
confusion matrix C(i,j) known to be in group i but predicted to be in group j
"""
if False:
    confusion = confusion_matrix(y_test, pred_logreg)
    print("Confusion matrix:\n{}".format(confusion))
    print("f1 score logistic regression: {:.2f}".format(
        f1_score(y_test, pred_logreg)))
    print(classification_report(y_test, pred_logreg,
                                target_names=["not nine", "nine"]))
"""
PRECISION_RECALL_CURVE
precision high -> recall high 阈值 平衡

"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)
# RandomForestClassifier has predict_proba, but not decision_function
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1])

plt.plot(precision, recall, label="svc")

close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero svc", fillstyle="none", c='k', mew=2)

plt.plot(precision_rf, recall_rf, label="rf")
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k',
         markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)

"""
Receiver operating characteristics (ROC)

instead of reporting precision and recall, it shows the false positive rate (FPR) against the true positive rate (TPR).
ROC 曲线（接收者操作特征曲线）是一种显示分类模型在所有分类阈值下的效果的图表
FPR（x轴） = FP/(FP+TN)
TPR = TP/(TP+FN)

在不同的阈值下TPR，FPR

you want a classifier that produces a high recall ->  low FPR



预测偏差 = avg(predict) - avg(对应标签)
造成预测偏差的可能原因包括：
    特征集不完整
    数据集混乱
    模型实现流水线中有错误？
    训练样本有偏差
    正则化过强


 
"""
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

"""
area under the curve AUC (summarize the ROC curve using a single number)
highly recommend using AUC when evaluating models on imbalanced data

"""
from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("AUC for Random Forest: {:.3f}".format(rf_auc))
print("AUC for SVC: {:.3f}".format(svc_auc))

"""
The most commonly used metric for imbalanced datasets in the multiclass setting 
is the multiclass version of the f-score
"""
from sklearn.metrics import accuracy_score

if False:
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=0)
    lr = LogisticRegression().fit(X_train, y_train)
    pred = lr.predict(X_test)
    print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))
    print(classification_report(y_test, pred))
"""
"macro" 
    averaging computes the unweighted per-class f-scores. This gives equal weight to all classes, no matter what their size is.
"weighted" 
    averaging computes the mean of the per-class f-scores, weighted by their support. This is what is reported in the classification report.
"micro" 
    averaging computes the total number of false positives, false negatives, and true positives over all classes, 
    and then computes precision, recall, and f- score using these counts.

If you care about each sample equally much, it is recommended to use the "micro" average f1-score;
if you care about each class equally much, it is recommended to use the "macro" average f1-score
"""
if False:
    print("Micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="micro")))
    print("Macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="macro")))

"""
Regression Metrics used in the score method of all regressors is enough.
"""

"""
Model Selection

"""
if False:
    # default scoring for classification is accuracy
    print("Default scoring: {}".format(
        cross_val_score(SVC(), digits.data, digits.target == 9)))
    # providing scoring="accuracy" doesn't change the results
    explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9,
                                        scoring="accuracy")
    print("Explicit accuracy scoring: {}".format(explicit_accuracy))
    roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9,
                              scoring="roc_auc")
    print("AUC scoring: {}".format(roc_auc))
"""
GridSearchCV

scoring parameter for classification are accuracy (the default); 
roc_auc for the area under the ROC curve; 
average_precision for the area under the precision-recall curve; 
f1, f1_macro, f1_micro, and f1_weighted for the binary f1-score and the different weighted variants.

For regression, the most com‐ monly used values are r2 for the R2 score
"""
if False:
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target == 9, random_state=0)
    # we provide a somewhat bad grid to illustrate the point:
    param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
    # using the default scoring of accuracy:

    grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
    grid = GridSearchCV(SVC(), param_grid=param_grid)

    grid.fit(X_train, y_train)
    print("Grid-Search with accuracy")
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
    print("Test set AUC: {:.3f}".format(
        roc_auc_score(y_test, grid.decision_function(X_test))))
    print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
