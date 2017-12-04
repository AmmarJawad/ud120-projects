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
# Removing the 'TOTAL' value in data_dict because it is a column sum of salaries and doesn't belong to any single employee.
del data_dict['TOTAL']

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

# Provided to give you a starting point. Try a variety of classifiers.
clf = AdaBoostClassifier(random_state=42)

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

### Create new feature(s)
# New feature is bonus/salary and null-values replaced by 0
features_train['bonus_salary_ratio'] = features_train.loc[:, 5] / features_train.loc[:, 1]
features_train['bonus_salary_ratio'] = np.nan_to_num(features_train['bonus_salary_ratio'])

# Repeat same feature engineering for test data
features_test['bonus_salary_ratio'] = features_test.loc[:, 5] / features_test.loc[:, 1]
features_test['bonus_salary_ratio'] = np.nan_to_num(features_test['bonus_salary_ratio'])


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
