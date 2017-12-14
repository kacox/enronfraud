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


def replace_nans(array, replacement):
    """
    Takes an array and a replacement value.

    Any NaN elements are replaced with replacement.

    Returns the resulting array as a list.
    """
    array_copy = list(array)
    for indx, element in enumerate(array):
        if element == "NaN":
            array_copy[indx] = replacement

    return array_copy


def make_2D_plot(data_as_dict, feature_x, feature_y):
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

def create_fraction_feature(data_as_dict, num_feature, denom_feature, 
                            new_feature):
    """
    Takes the Enron data as a dictionary (data_as_dict), a feature to serve 
    as the numerator, a feature to serve as the denominator, and the desired 
    name for the newly created feature; the latter three items are strings.
    
    Returns the input dictionary with the addition of the new feature.
    """
    for person in data_as_dict.iterkeys():
        if (data_as_dict[person][num_feature] == "NaN") or \
                            (data_as_dict[person][denom_feature] == "NaN"):
            fraction_to_poi = 0
            data_as_dict[person][new_feature] = fraction_to_poi
        else:
            numerator = float(data_as_dict[person][num_feature])
            denominator =  float(data_as_dict[person][denom_feature])
            fraction_to_poi = numerator / denominator
            data_as_dict[person][new_feature] = fraction_to_poi
            
    return data_as_dict


def rescale_features(data_as_dict, feature_list):
    """
    Takes the Enron data as a dictionary (data_as_dict) and a list of features 
    to rescale (normalize).
    
    Returns a dictionary with the desired features rescaled.
    """
    # Make copy of input dictionary
    rescaled_dict = data_as_dict.copy()
    
    # Rescale feature by feature
    for feature in feature_list:
        val_list = []
        person_list = []
        
        # Build lists in preserved order with persons and respective values 
        # for the given iteration's feature
        for person in data_as_dict.iterkeys():
            person_list.append(person)
            val_list.append(data_as_dict[person][feature])
            
        # Replace NaNs in val_list
        val_list = replace_nans(val_list, 0)
        
        # Rescale values
        xmin = min(val_list)
        xmax = max(val_list)        
        rescaled_val = []
        for val in val_list:
            rescaled = float((val - xmin))/(xmax - xmin)
            rescaled_val.append(rescaled)

        
        # Put rescaled value into dict copy
        for person in data_as_dict.iterkeys():
            rescaled_dict[person][feature] = \
                                rescaled_val[person_list.index(person)]
                                       
    return rescaled_dict