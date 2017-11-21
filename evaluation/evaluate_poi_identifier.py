

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(pred, labels_test)


counter = 0
for label in pred:
    if label == 1.:
        counter += 1
print "Pois predicted: ", counter
print "Total people in test set: ", len(labels_test)
print "If identifier predicted 0.0 (not POI) for everyone in the test set, what would the accuracy be?", (float(len(labels_test))-float(counter))/float(len(labels_test))
print "precision_score: ", precision_score(pred, labels_test)
print "recall_score: ", recall_score(pred, labels_test)
