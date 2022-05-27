


# Dependencies 
import os 
import pandas as pd
import numpy as np

from typing import Callable, Tuple, Dict, List 

from  sdg_utils import Positive, Binary

def prepare_PIMA(dataset_path : str = "", filename : str = "") :
    """Read the PIMA dataset from a .csv file and suit it tu be processed 
    as a pd.DataFrame. It returns tha dataset dataframe and strings associated to 
    it to easy its management.

    Args:
    -----
            dataset_path: path where PIMA dataset is stored. Set by default.
            filename : file name of the .csv containing the dataset. Set by default.

    Returns:
    --------
            data: dataframe containing the whole dataset
            X : dataframe containing the dataset features
            Y : dataframe containing only the target variable
            cols_names: list of strings containing feature names. 
            y_tag: string containing target variable name.
    """

    # Go to dataset path
    os.chdir(dataset_path)

    # Load data into a DataFrame
    data = pd.read_csv(filename)

    # Store column names 
    cols_names = data.columns
    
    # Cleaning data phase:
    # Zero-values are considered missing values except from the Pregnancies variable
    data['Glucose']  = data['Glucose'].replace([0],[np.nan])
    data['BloodPressure']  = data['BloodPressure'].replace([0],[np.nan])
    data['SkinThickness']  = data['SkinThickness'].replace([0],[np.nan])
    data['Insulin']  = data['Insulin'].replace([0],[np.nan])
    data['BMI']  = data['BMI'].replace([0],[np.nan])
    data['DiabetesPedigreeFunction']  = data['DiabetesPedigreeFunction'].replace([0],[np.nan])
    data['Age']  = data['BMI'].replace([0],[np.nan])
    
    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

def numerical_conversion(data : np.array, features : str, y_col : str):
    """Fix all PIMA database features data types to its original type after KNNImputer is used,
    since this functions returns only a floating points ndarray. For more, check sklearn 
    documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. After 
    fixing datatypes, an ndarray tonp.DataFrame conversion is performed. Notice that this
    operation is only done in the fields that were not originally floats.

    Args:
    -----
            data: data returned by KNN Imputer (float data type).
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe containing the whole dataset after imputation
            X : dataframe containing the dataset features after imputation
            y : dataframe containing only the target variable after imputation
    """
    # From ndarray to pd.DataFrame
    names = features.insert(len(features), y_col)
    data = pd.DataFrame(data, columns = names)
    
    # Fixing necessary datatypes to int 
    data['Pregnancies'] = data['Pregnancies'].astype(int)
    data['Glucose'] = data['Glucose'].astype(int)
    data['BloodPressure'] = data['BloodPressure'].astype(int)
    data['SkinThickness'] = data['SkinThickness'].astype(int)
    data['Insulin'] = data['Insulin'].astype(int)
    data['Age'] = data['Age'].astype(int)
    data['Outcome'] = data['Outcome'].astype(int)

    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

# Dictionary to specify fields of synthetic data for PIMA database
pima_fields = {
    'Pregnancies' : {
        'type' : 'numerical',
        'subtype' : 'integer',
    },
    'Glucose' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'BloodPressure' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'SkinThickness' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Insulin' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'BMI' : {
        'type' : 'numerical',
        'subtype' : 'float'
    }, 
    'DiabetesPedigreeFunction' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Outcome' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }     
 }

# Custom variable constraints to generate synthetic data 
constraints = [ 
                Positive('Pregnancies',handling_strategy='reject_sampling'),
                Positive('Glucose',handling_strategy='reject_sampling'),
                Positive('BloodPressure',handling_strategy='reject_sampling'),
                Positive('SkinThickness',handling_strategy='reject_sampling'), 
                Positive('Insulin',handling_strategy='reject_sampling'),
                Positive('BMI',handling_strategy='reject_sampling'),
                Positive('DiabetesPedigreeFunction',handling_strategy='reject_sampling'),
                Positive('Age',handling_strategy='reject_sampling'),
                Binary('Outcome',handling_strategy='reject_sampling')
               ]

# Distributions for each field (all set to univariate)
pima_distributions = {
    'Pregnancies' : 'univariate', #'gamma',
    'Glucose' : 'univariate', #'gaussian',
    'BloodPressure' : 'univariate', #'gaussian',
    'SkinThickness' : 'univariate', #'gaussian',
    'Insulin' : 'univariate', #'gamma',
    'BMI' : 'univariate', #'gaussian',
    'DiabetesPedigreeFunction' : 'univariate', #'gamma',
    'Age' : 'univariate', #'gamma',
    'Outcome' : 'univariate', #'gaussian',
    }