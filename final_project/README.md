### enronfraud
# Repository for Identifying Fraud from Enron Email

## Overview
This project details the investigation and use of a machine learning algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

The dataset is converted to a dictionary (`data_dict`) containing 146 employees and 21 features.

**Question guiding investigation:** Which Enron employees are persons of interest (POI)?

## Main Files  
**poi_id.py** - Main Python script file that contains my analysis and final algorithm choice. Automatically creates `my_classifier.pkl`, `my_dataset.pkl`, and `my_feature_list.pkl` for use by `tester.py`.  
**helper_fxns.py** - Python script containing auxiliary functions used in `poi_id.py`. 
**questions.docx** - Document containing questions and answers meant for reflection and project explanations.  
**tester.py** - Provided Python script from Udacity that evaluates the algorithm created in `poi_id.py`.   
**references.txt** - References used in creating this project. 
