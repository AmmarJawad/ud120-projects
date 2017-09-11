#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
def svmclf(features_train, labels_train):
    from sklearn.svm import SVC
    clf = SVC(C=10000, kernel="rbf")

    # One way to speed up an algorithm is to train it on a smaller training
    # dataset. The tradeoff is that the accuracy almost always goes down
    # when you do this.
    # features_train = features_train[:len(features_train)/100]
    # labels_train = labels_train[:len(labels_train)/100]

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print "Overall accuracy", acc
    print "Element 10: ", pred[10]
    print "Element 26:", pred[26]
    print "Element 50:", pred[50]

    # counting how many emails were predicted to be from either Chris (1) or
    # Sara (0)
    Sara = 0
    Chris = 0
    for i in pred:
        if i == 1:
            Chris += 1
    Sara = len(pred) - Chris
    print "Chris: ", Chris
    print "Sara: ", Sara


svmclf(features_train, labels_train)
