#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

# data_dict.pop removes the entry "TOTAL" and replace it with 0
data_dict.pop("TOTAL", 0)

def who_are_the_outliers():
    '''Find Enron employees who earn more than 1m per year and more than 5m in bonus'''
    for key in data_dict.keys():
        if data_dict[key]['bonus'] > 5000000 and data_dict[key]['salary'] > 1000000:
            if data_dict[key]['bonus'] != "NaN" and data_dict[key]['salary'] != 'NaN':
                print "Name of outlier: ", key

who_are_the_outliers()
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
