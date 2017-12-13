#!/usr/bin/python


### IMPORTS AND GLOBALS
import sys
import pickle
sys.path.append("../tools/")
import numpy as np

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
#helper.draw_1D(data_dict, "director_fees")
#helper.draw_1D(data_dict, "deferred_income")
#helper.draw_1D(data_dict, "restricted_stock_deferred")
#helper.draw_1D(data_dict, "loan_advances")
#helper.draw_1D(data_dict, "deferral_payments")

# Remove features with too many missing values
removal_list = ["director_fees", "deferred_income", 
                "restricted_stock_deferred", "loan_advances", 
                "deferral_payments", "email_address"]

my_features_list = list(features_list)
for item in removal_list:
    my_features_list.remove(item) 


### Task 2: Remove outliers
# Remove artifact in data (found during lectures)
data_dict.pop("TOTAL", 0)

# Visual inspection
helper.draw(data_dict, "salary", "bonus")
helper.draw_1D(data_dict, "salary")
#print helper.find_max_person(data_dict, "salary")

# Check data points with many missing values
nan_dict = {}
for person, feature_dict in data_dict.iteritems():
    for feature in feature_dict.iterkeys():
        if feature_dict[feature] == "NaN":
            if person in nan_dict.keys():
                nan_dict[person] += 1
            else:
                nan_dict[person] = 1

#print sorted(nan_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
#print data_dict['LOCKHART EUGENE E']
#print data_dict['THE TRAVEL AGENCY IN THE PARK']

# Remove these data points
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


### Task 3: Create new feature(s)
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

### Extract features and labels from dataset for local testing
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
print X_df.columns[selector.get_support(indices=True)]
print sorted(selector.scores_, reverse=True)[:5]


### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

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


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# K-folds cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def kfold_eval(clf, X_train, y_train, X_test, y_test, num_folds):
    """
    Takes a classifier object, training features, training labels, test 
    features, test labels, and the number of folds (independent experiments) 
    to perform.
    
    Reports the average scores, recall, and precision for the given 
    classifier.
    
    Returns nothing.
    """
    cumulative_validation = []
    cumulative_test_score = []
    cumulative_recall = []
    cumulative_precision = []
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    for train_indices, validation_indices in kf.split(X_train):
        # Designate training and validation sets
        features_train = [X_train[t_indx] for t_indx in train_indices]
        labels_train = [y_train[t_indx] for t_indx in train_indices]
        features_validation = [X_train[v_indx] for v_indx in validation_indices]
        labels_validation = [y_train[v_indx] for v_indx in validation_indices]
    
        # Fit
        clf.fit(features_train, labels_train)
        
        # Score using validation set
        cumulative_validation.append(clf.score(features_validation, labels_validation))
        
        # Score with X_test, y_test (withheld from validation)
        cumulative_test_score.append(clf.score(X_test, y_test))
        cumulative_recall.append(recall_score(y_test, 
                                                  clf.predict(X_test)))
        cumulative_precision.append(precision_score(y_test, 
                                                     clf.predict(X_test)))
        
    # Report averages
    print clf
    print "Avg. Validation Set Score:", (float(sum(cumulative_validation)) / 
                                         len(cumulative_validation))
    print "Avg. Test Set Score:", (float(sum(cumulative_test_score)) / 
                                   len(cumulative_test_score))
    print "Avg. Recall:", (float(sum(cumulative_recall)) / len(cumulative_recall))
    print "Avg. Precision:", (float(sum(cumulative_precision)) / 
                                         len(cumulative_precision))


    
clf = DecisionTreeClassifier(min_samples_split=5)  
kfold_eval(clf, X_train, y_train, X_test, y_test, 4)
"""
# GridSearchCV (cross validation for parameter tuning)
from sklearn.model_selection import GridSearchCV
clf_to_tune = DecisionTreeClassifier()
parameters = {'min_samples_split':(2, 3, 4, 5, 6)}
cv_clf = GridSearchCV(clf, parameters)
cv_clf.fit(X_train, y_train)
print cv_clf.best_params_
"""


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)

