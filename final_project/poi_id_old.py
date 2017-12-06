#!/usr/bin/python


## IMPORTS AND GLOBALS
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


## FUNCTIONS
def create_feature_arrays(data_as_dict, feature_names):
    """
    Takes the Enron data as a dictionary (data_dict) and a list of the 
    desired feature names as ['poi', 'feature1', 'feature2'].

    Returns two arrays, each containing the values of their respective 
    features.
    """
    feature1_list, feature2_list = [], []
    for info in data_as_dict.values():
        for feature, val in info.items():
            if feature == feature_names[1]:
                if val == "NaN":
                    feature1_list.append(np.nan)
                else:
                    feature1_list.append(val)
            elif feature == feature_names[2]:
                if val == "NaN":
                    feature2_list.append(np.nan)
                else:
                    feature2_list.append(val)

    return feature1_list, feature2_list


def replace_nans(array, replacement):
    """
    Takes an array and a replacement value.

    Any NaN elements are replaced with replacement.

    Returns the resulting array as a list.
    """
    array_copy = list(array)
    for indx, element in enumerate(array):
        if np.isnan(element):
            array_copy[indx] = replacement

    return array_copy


def draw(data_as_dict, feature_names, mark_poi=False):
    """ 
    Takes the Enron data as a dictionary (data_dict), a list of the feature
    names as ['poi', 'feature1', 'feature2'], and whether or not the 
    data points with 'poi'=True should be marked in the plot.

    Creates a scatterplot; nothing is returned.
    """

    # create feature arrays
    feature1_list, feature2_list = create_feature_arrays(data_as_dict, feature_names)

    # plot
    plt.scatter(feature1_list, feature2_list, color = "b")
    plt.xlabel(feature_names[1])
    plt.ylabel(feature_names[2])
    plt.show()


def find_outliers(data_as_dict, feature_names, percentile):
    """
    Takes the Enron data as a dictionary (data_dict), a list of the feature
    names as ['poi', 'feature1', 'feature2'], and an integer indicating 
    the percentile above which data points should be considered outliers.

    Returns outlier thresholds for feature1 and feature2.  
    """
    # create feature arrays
    feature1_list, feature2_list = create_feature_arrays(data_as_dict, feature_names)

    # replace NaNs with 0
    no_nans_feature1 = replace_nans(feature1_list, 0)
    no_nans_feature2 = replace_nans(feature2_list, 0)

    # find cutoff
    feature1_threshold = np.percentile(no_nans_feature1, percentile)
    feature2_threshold = np.percentile(no_nans_feature2, percentile)

    return feature1_threshold, feature2_threshold


def remove_outliers(data_dict, feature_names, percentile):
    """
    """
    data_dict_copy = dict(data_dict)

    # find outliers
    threshold1, threshold2 = find_outliers(data_dict, feature_names, percentile)

    # remove outliers
    for person, feature_dict in data_dict.iteritems():
        removed_flag = False
        for feature, val in feature_dict.iteritems():
            if removed_flag == False:
                if (feature == feature_names[1]) and (val > threshold1):
                    data_dict_copy.pop(person)
                    removed_flag = True
                elif (feature == feature_names[2]) and (val > threshold2):
                    data_dict_copy.pop(person)
                    removed_flag = True

    return data_dict_copy


## MAIN
"""
Task 1: Select what features you'll use.
features_list is a list of strings, each of which is a feature name.
The first feature must be "poi".
"""
features_list = ['poi','salary', 'expenses'] # You will need to use more features
test_features_list = ['poi','salary', 'expenses']   # visual inspection

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Draw to visually check features and correlations
cleaned_data_dict = remove_outliers(data_dict, test_features_list, 99)
draw(cleaned_data_dict, test_features_list)

"""
# Checking data_dict structure and contents
feature_count = 0
for key, val in data_dict.items():
    for feature in val:
        feature_count += 1
    break

print feature_count
"""

"""
### Task 2: Remove outliers
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
