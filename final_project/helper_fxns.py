"""
Helper functions for `poi_id.py` project file.
"""
import numpy as np
import matplotlib.pyplot as plt


def count_missing_values(data_as_dict):
    """
    Takes the Enron data as a dictionary (data_as_dict) and counts the number 
    of missing values for every feature.
    
    Returns a dictionary containing features and the number of missing values 
    for each:
        {feature_a : 20, feature_b : 44, ..., feature_z : 12}
    """
    missing_value_dict = {}
    
    for info in data_as_dict.itervalues():
        for feature, val in info.iteritems():
            if val == "NaN":
                if feature not in missing_value_dict.keys():
                    missing_value_dict[feature] = 1
                else:
                    missing_value_dict[feature] += 1
            
    return missing_value_dict

def draw_1D(data_as_dict, feature_name):
    """
    Takes the Enron data as a dictionary (data_as_dict) and a feature to plot 
    as a string.

    Creates a 1D scatterplot; nothing is returned.
    """
    feature_x_list = create_feature_array(data_as_dict, feature_name)
    poi_list = create_feature_array(data_as_dict, "poi")

    # split into poi and non-poi arrays
    poi_true = []
    poi_false = []
    for indx, feature_value in enumerate(feature_x_list):
        if poi_list[indx] == True:
            poi_true.append(feature_x_list[indx])
        elif poi_list[indx] == False:
            poi_false.append(feature_x_list[indx])
            
    # plot
    plt.plot(poi_false, np.zeros_like(poi_false), 'bo')
    plt.plot(poi_true, np.zeros_like(poi_true), 'rx')
    plt.xlabel(feature_name)
    plt.show()


def remove_outliers(data_dict, feature1, feature2, percentile):
    """
    Takes the Enron data as a dictionary (data_dict), two feature names as 
    strings, and a percentile beyond which data points are considered 
    outliers.
    
    Removes outlier data points.
    
    Returns a a copy of the input dictionary without outlier data points.
    """
    data_dict_copy = dict(data_dict)

    # find outliers
    threshold1, threshold2 = find_outliers(data_dict, feature1, feature2, 
                                           percentile)

    # remove outliers
    for person, feature_dict in data_dict.iteritems():
        removed_flag = False
        for feature, val in feature_dict.iteritems():
            if removed_flag == False:
                if (feature == feature1) and (val > threshold1):
                    data_dict_copy.pop(person)
                    removed_flag = True
                elif (feature == feature2) and (val > threshold2):
                    data_dict_copy.pop(person)
                    removed_flag = True

    return data_dict_copy


def find_outliers(data_as_dict, feature1, feature2, percentile):
    """
    Takes the Enron data as a dictionary (data_dict), two feature names as 
    strings, and an integer indicating the percentile above which data points 
    should be considered outliers.

    Returns outlier thresholds for feature1 and feature2.  
    """
    # create feature arrays
    feature1_list = create_feature_array(data_as_dict, feature1)
    feature2_list = create_feature_array(data_as_dict, feature2)

    # replace NaNs with 0
    no_nans_feature1 = replace_nans(feature1_list, 0)
    no_nans_feature2 = replace_nans(feature2_list, 0)

    # find cutoff
    feature1_threshold = np.percentile(no_nans_feature1, percentile)
    feature2_threshold = np.percentile(no_nans_feature2, percentile)

    return feature1_threshold, feature2_threshold


def replace_nans(array, replacement):
    """
    Takes an array and a replacement value.

    Any NaN elements are replaced with replacement.

    Returns the resulting array as a list.
    """
    import numpy as np
    
    array_copy = list(array)
    for indx, element in enumerate(array):
        if np.isnan(element):
            array_copy[indx] = replacement

    return array_copy


def draw(data_as_dict, feature_x, feature_y):
    """ 
    Takes the Enron data as a dictionary (data_as_dict) and two features to 
    plot as strings.

    Creates a scatterplot; nothing is returned.
    """    
    # create feature arrays
    feature_x_list = create_feature_array(data_as_dict, feature_x)
    feature_y_list = create_feature_array(data_as_dict, feature_y)
    poi_list = create_feature_array(data_as_dict, "poi")

    # create color list for plotting
    color_list = ['r' if outcome == True else 'b' for outcome in 
                   poi_list]

    # plot
    plt.scatter(feature_x_list, feature_y_list, c=color_list)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


def create_feature_array(data_as_dict, feature_name):
    """
    Takes the Enron data as a dictionary (data_as_dict) and a feature as a 
    string.

    Returns an array containing the values of the respective feature.
    """    
    feature_list = []
    for info in data_as_dict.values():
        for feature, val in info.items():
            if feature == feature_name:
                if val == "NaN":
                    feature_list.append(np.nan)
                else:
                    feature_list.append(val)

    return np.array(feature_list)


def find_max_person(data_as_dict, feature_name):
    """
    Takes the Enron data as a dictionary (data_as_dict) and a feature as a 
    string.

    Returns the person with the highest value for that feature and the value
    itself.
    """  
    max_val, max_person = float("-inf"), None
    for person, feature_dict in data_as_dict.iteritems():
        if (feature_dict[feature_name] != "NaN") and \
                                    (feature_dict[feature_name] > max_val):
            max_val = feature_dict[feature_name]
            max_person = person
    return max_person, max_val

