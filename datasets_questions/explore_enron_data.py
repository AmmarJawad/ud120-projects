#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


def feature_counter(enron_data):
    '''Counts the amount of features per Enron employee'''
    for key in enron_data.iterkeys():
        print "Employee name: ", key
        print "No. of features for employee: ", len(enron_data[key])
        print "\n"


feature_counter(enron_data)


def poi_counter(enron_data):
    '''Counts the amounts of Enron employees which are POIs'''
    pois = 0
    for employee in enron_data:
        if enron_data[employee]['poi']:
            pois += 1
    print "No. of employees that are POIs: ", pois


poi_counter(enron_data)
