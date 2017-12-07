#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import scipy

import missingno as msno
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred',
                'deferred_income', 'total_stock_value', 'expenses'
               ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Removing keys 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'BANNANTINE JAMES M' and 'GRAY RODNEY' in data_dict because they are either not employees or in the case of Gray and Bannantine they are outliers.
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['BANNANTINE JAMES M']
del data_dict['GRAY RODNEY']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# Transforming features into a df so that I won't have to remember to transform both features train and test.
df_features = pd.DataFrame(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(df_features, labels, test_size=0.3, random_state=42)



clf = GradientBoostingClassifier(random_state=42,
                                 min_samples_leaf=6,
                                 min_samples_split=20,
                                 n_estimators=98,
                                 max_features=5,
                                 max_depth=5
                                )
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# Classifier scores
f1_score_gbm = f1_score(pred, labels_test)
precision_score_gbm = precision_score(pred, labels_test, average='weighted')
recall_score_gbm = recall_score(pred, labels_test, average='weighted')
print "F1-score", f1_score_gbm
print "Precision score: ", precision_score_gbm
print "Recall score: ", recall_score_gbm

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
