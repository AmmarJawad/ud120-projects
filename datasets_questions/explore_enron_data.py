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


def pois_in_txtfile():
    '''Counts the total rows in the file'''
    count = 0
    with open('/Users/ammardoosh/Documents/GitHub2/UD120_MachineLearning/final_project/poi_names.txt', "r") as f:
        for line in f:
            if "(y)" in line or "(n)" in line:
                count += 1
    print "Length of rows in txt file", count


pois_in_txtfile()


def actual_pois():
    '''Counts only the rows in the file which contains (y)'''
    count = 0
    with open('/Users/ammardoosh/Documents/GitHub2/UD120_MachineLearning/final_project/poi_names.txt', "r") as f:
        for line in f:
            if '(y)' in line:
                count += 1
    print "POIs in existence: ", count


actual_pois()


def people_info():
    JPBonus = enron_data['PRENTICE JAMES']['total_stock_value']
    print "Total stock value of James Prentice: ", JPBonus

    WCemails = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
    print "Emails by Wesley Colwell: ", WCemails

    JFstock = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
    print "Value of stock options by Jeffrey K. Skilling: ", JFstock
    # print enron_data['SKILLING JEFFREY K']


people_info()


def employees_with_salaries():
    count_salary = 0
    count_emails = 0
    nan_total_payments = 0
    total_people = 0
    pois_with_nan = 0
    pois = 0
    for key in enron_data.keys():
        if enron_data[key]['salary'] != 'NaN':
            count_salary += 1
        if enron_data[key]['email_address'] != 'NaN':
            count_emails += 1
        if enron_data[key]['total_payments'] == 'NaN':
            nan_total_payments += 1
        if enron_data[key]:
            total_people += 1
        if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] is True:
            pois_with_nan += 1
        if enron_data[key]['poi'] is True:
            pois += 1
    print "Sum of employees with known salaries: ", count_salary
    print "Sum of employees with known email addresses: ", count_emails
    print "Sum of employees with NaN in total payments: ", nan_total_payments
    print "Total employees in dataset: ", total_people
    print "POIs with NaN in total payments: ", pois_with_nan
    print "POIs: ", pois


employees_with_salaries()
