import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import missingno
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBo, RandomForestClassifier, AdaBoostClassifieroForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#############################################
### Task 1: Select what features you'll use.
#############################################

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_stock_value']

# Identifying columns with financial values
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

# Identfying columns with numerical values
features_with_count = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###########################
### Task 2: Remove outliers
###########################

# Removing the 'TOTAL' value in data_dict because it is a sum and doesn't belong to any single employee for whom we are
# attempting to predict whether or not they are a poi.
# I don't want to remove other outliers because of the scarcity of data already.
del data_dict['TOTAL']

#################################
### Task 3: Create new feature(s)
#################################

### Store to my_dataset for easy export below.
#print data_dict
my_dataset = data_dict
#print my_dataset
### Extract features and labels from dataset for local testing
data =t(my_dataset, features_list, sort_keys = Truecolumns = ['salary', 'total_stock_value']
print df_features.head()



########################################
### Task 4: Try a varity of classifiers
########################################

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelifo:
##AdaBoostClassifiercrandom_state=42i.lf
clf = GaussianNB()
#clf = GradientBoostingClassifier()


#clf = GaussianNB()

###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
###############################################################################


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_s   'min_samples_leaf':np.linspace(1red, labels_test)
recall_score = recall_score(pred, labels_test)
accuracy_score = accuracy_score(pred, labels_test)

###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessae
print "Precision score: ", precision_score
print "Recll score: ", recall_score

# print confusion_matrix(labels_test, pred)
dump_classifier_and_data(clf, my_dataset, features_list)
