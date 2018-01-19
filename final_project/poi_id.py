#!/usr/bin/python


### IMPORTS AND GLOBALS
import sys
import pickle
import operator
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### FUNCTIONS AND DEPENDENCIES
import helper_fxns as helper


### MAIN

"""
TASK 1: Select what features you'll use.
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


# Remove features with too many missing values
removal_list = ["director_fees", "deferred_income", 
                "restricted_stock_deferred", "loan_advances", 
                "deferral_payments", "email_address"]

my_features_list = list(features_list)
for item in removal_list:
    my_features_list.remove(item) 


"""
TASK 2: Remove outliers
"""

# Remove artifact in data (found during lectures)
data_dict.pop("TOTAL", 0)

# Visual inspection
helper.make_2D_plot(data_dict, "salary", "bonus")
helper.draw_1D(data_dict, "salary")
print helper.find_max_person(data_dict, "salary")

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
TASK 3: Create new feature(s)
"""

# Engineer new features from old
data_dict = helper.create_fraction_feature(data_dict,"from_this_person_to_poi", 
                                           "from_messages", "fraction_to_poi")
data_dict = helper.create_fraction_feature(data_dict,"from_poi_to_this_person", 
                                           "to_messages", "fraction_from_poi")

# Add new features to list
print len(my_features_list)
my_features_list.append("fraction_to_poi")
my_features_list.append("fraction_from_poi")
print len(my_features_list)

# Normalize features
no_poi_features_list = list(my_features_list)
no_poi_features_list.remove("poi")
my_dataset = helper.rescale_features(data_dict, no_poi_features_list)

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Select k best features
from sklearn.feature_selection import SelectKBest
import pandas as pd
X_df = pd.DataFrame(features, columns=no_poi_features_list) 
y_df = pd.DataFrame(labels) 

print "before:", X_df.shape
selector = SelectKBest(k=5)
selector.fit(X_df, y_df)
X_new = selector.transform(X_df)
print "after:", X_new.shape

# Get names of K best features
k_best_features = (["poi"] + 
                   (list(X_df.columns[selector.get_support(indices=True)])))
print sorted(selector.scores_, reverse=True)[:5]

# Extract features and labels using K best features
data = featureFormat(my_dataset, k_best_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


"""
TASK 4: Try a variety of classifiers

Please name your classifier clf for easy export below. Note that if you want 
to do PCA or other multi-stage operations, you'll need to use Pipelines. For 
more info: http://scikit-learn.org/stable/modules/pipeline.html
"""

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                    test_size=0.3, random_state=81)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
#print nb_clf.score(X_test, y_test)
# 0.7907 (random_state=81)
# 0.9302 (random_state=37)

# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
#print dt_clf.score(X_test, y_test)
# 0.8605 (random_state=81)
# 0.8605 (random_state=37)

# Support Vector Machine
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)
#print svm_clf.score(X_test, y_test)
# 0.9070 (random_state=81)
# 0.9767 (random_state=37)


"""
TASK 5: Tune your classifier to achieve better than .3 precision and recall 
using our testing script. Check the tester.py script in the final project
folder for details on the evaluation method, especially the test_classifier
function. Because of the small size of the dataset, the script uses
stratified shuffle split cross validation. For more info: 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
"""

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

print nb_clf
print "Validation Set Score:", nb_clf.score(X_test, y_test)
print "Recall:", recall_score(y_test, nb_clf.predict(X_test))
print "Precision:", precision_score(y_test, nb_clf.predict(X_test))

"""
# GridSearchCV (cross validation for parameter tuning)
from sklearn.model_selection import GridSearchCV
clf_to_tune = DecisionTreeClassifier()
parameters = {'min_samples_split':(2, 3, 4, 5, 6)}
cv_clf = GridSearchCV(clf, parameters)
cv_clf.fit(X_train, y_train)
print cv_clf.best_params_
"""


"""
TASK 6: Dump your classifier, dataset, and features_list so anyone can check 
your results. You do not need to change anything below, but make sure that 
the version of poi_id.py that you submit can be run on its own and generates 
the necessary .pkl files for validating your results.
"""

# Final choice features list and algorithm.
my_features_list = k_best_features
clf = GaussianNB()

#dump_classifier_and_data(clf, my_dataset, my_features_list)

