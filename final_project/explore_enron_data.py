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
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# 146 data points (people)

"""
poi_count = 0
for person, features in enron_data.iteritems():
    if features["poi"] == True:
        poi_count += 1

print poi_count # 18 P.O.I. in dictionary, though 35 in total exist (see names txt file)
"""

# highest $$
highest_salary = 0
highest_name = None
for person, features in enron_data.iteritems():
    if (features['salary'] != "NaN") and (features['salary'] > highest_salary):
        highest_salary = features['salary']
        highest_name = person

print highest_name, highest_salary


"""
# How many people have quantified salaries in this dataset?
sal_count = 0
for person, features in enron_data.iteritems():
    if features['salary'] == "NaN":
        enron_data[person]['salary'] = np.nan
    else:
        sal_count += 1

print sal_count
"""

"""
email_count = 0
for person, features in enron_data.iteritems():
    if features['email_address'] != "NaN":
        email_count += 1

print email_count
"""

"""
# How many people in the dataset (as it currently exists) have "NaN" for their total payments? 
# What percentage of people in the dataset as a whole is this?
tp_nan_count = 0
total = 0
for person, features in enron_data.iteritems():
    if features['total_payments'] == "NaN":
        tp_nan_count += 1
        total += 1
    else:
        total += 1

print total, tp_nan_count
print (float(tp_nan_count) / total) * 100, "%"  # 14%
"""

"""
# How many POIs in the E+F dataset have "NaN" for their total payments? What percentage of POIs as a whole is this?
poi_tp_nan_count = 0
poi_total = 0
for person, features in enron_data.iteritems():
    if features['poi'] == True:
        if features['total_payments'] == "NaN":
            poi_tp_nan_count += 1
            poi_total += 1
        else:
            poi_total += 1

print poi_total, poi_tp_nan_count       # 18, 0
print (float(poi_tp_nan_count) / poi_total) * 100, "%"      # 0%
"""
