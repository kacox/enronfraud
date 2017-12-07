#!/usr/bin/python


### IMPORTS AND GLOBALS
import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import operator

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### FUNCTIONS AND DEPENDENCIES
import helper_fxns as helper


### MAIN
"""
Task 1: Select what features you'll use.
"""
# boolean, represented as integer
POI_label = ['poi']

# all units are in US dollars
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock', 
                      'director_fees']

# units are generally number of emails messages; notable exception is 
# 'email_address', which is a text string
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']

# full list, POI_label first
features_list = POI_label + financial_features + email_features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

"""
# Checking data_dict structure
feature_count = 0
feature_count_flag = False
poi_count = 0
for person, features_dict in data_dict.items():
    for feature, val in features_dict.iteritems():
        if feature_count_flag == False:
            feature_count += 1
        if (feature == "poi") and (val == True):
            poi_count += 1
    feature_count_flag = True

print "No. data pts:", len(data_dict)
print "No. features:", feature_count
print "No. POI:", poi_count
print "No. non-POI:", (len(data_dict) - poi_count)
"""

# Inspecting features for missing values
nan_dict = helper.count_missing_values(data_dict)
nan_list = [val for val in nan_dict.values()]
print nan_dict
print "Mean:", np.mean(nan_list)
print "Median:", np.median(nan_list)
print "75th percentile:", np.percentile(nan_list, 75)

# 1D inspection of missing values (checking before ruling out)
helper.draw_1D(data_dict, "director_fees")
helper.draw_1D(data_dict, "deferred_income")
helper.draw_1D(data_dict, "restricted_stock_deferred")
helper.draw_1D(data_dict, "loan_advances")
helper.draw_1D(data_dict, "deferral_payments")

# Remove features with too many missing values
removal_list = ["director_fees", "deferred_income", 
                "restricted_stock_deferred", "loan_advances", 
                "deferral_payments"]

my_features_list = list(features_list)
for item in removal_list:
    my_features_list.remove(item) 


### Task 2: Remove outliers
# Remove artifact in data (found during lectures)
data_dict.pop("TOTAL", 0)

# Visual inspection
helper.draw(data_dict, "salary", "other")
helper.draw_1D(data_dict, "other")
print helper.find_max_person(data_dict, "other")

# Check data points with many missing values
nan_dict = {}
for person, feature_dict in data_dict.iteritems():
    for feature in feature_dict.iterkeys():
        if feature_dict[feature] == "NaN":
            if person in nan_dict.keys():
                nan_dict[person] += 1
            else:
                nan_dict[person] = 1

print sorted(nan_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
print data_dict['LOCKHART EUGENE E']
print data_dict['THE TRAVEL AGENCY IN THE PARK']

# Remove these data points
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

"""
### Task 3: Create new feature(s)
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
"""
