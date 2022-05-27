# Dependencies 
import os 
import pandas as pd
import numpy as np

from  sdg_utils import Positive, Binary 

from typing import Tuple

from sklearn.preprocessing import OneHotEncoder

def prepare_SACardio(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the South Africa disease dataset from a .txt file and suit it to be processed 
    from .arff format to as a pd.DataFrame. Some transformations are applied to handle 
    this dataset properly (i.e., from "Present" to "1" and "Absent" to "0", etc. 
    This converted DataFrame is returned. 

    Args:
    -----
            dataset_path: path where dataset is stored. Set by default.
            filename : file name of the .arff containing the dataset. Set by default.

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

    # Open the .csv file and convert it into DataFrame
    data = pd.read_csv(filename, sep=";", header='infer')

    # Convert "Present" and "No" (and equivalent) values into "1" and "0" 
    data.replace(('Present', 'Absent'), (1, 0), inplace=True)

    # Drop ID column 
    data = data.drop(['ind'], axis=1)

    # Store column names 
    cols_names = data.columns

    # Store features' and target variable's names 
    cols_names_prev = data.columns
    y_tag = cols_names_prev[len(cols_names_prev)-1]
    cols_names = cols_names_prev[0:cols_names_prev.size]

    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

# Dictionary to specify fields of synthetic data for Alzheimer-Balea database
SACardio_fields = {
    'sbp' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'tobacco' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },  
    'ldl' : {
        'type' : 'numerical',
        'subtype' : 'float'
    }, 
    'adiposity' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'famhist' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'typea' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'obesity' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'alcohol' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'chd' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }                 
}

# Custom variable constraints to generate synthetic data 
constraints =constraints = [ 
                # Positive('Age',handling_strategy='reject_sampling'),              
                # Binary('Polyuria',handling_strategy='reject_sampling'),                                
                # Binary('Polyphagia',handling_strategy='reject_sampling'),
                # Binary('Polydipsia',handling_strategy='reject_sampling'), 
                # Binary('sudden weight loss',handling_strategy='reject_sampling'),
                # Binary('weakness',handling_strategy='reject_sampling'),
                # Binary('Genital thrush',handling_strategy='reject_sampling'),
                # Binary('visual blurring',handling_strategy='reject_sampling'),
                # Binary('Irritability',handling_strategy='reject_sampling'),
                # Binary('Itching',handling_strategy='reject_sampling'),
                # Binary('delayed healing',handling_strategy='reject_sampling'),
                # Binary('partial paresis',handling_strategy='reject_sampling'),
                # Binary('muscle stiffness',handling_strategy='reject_sampling'),
                # Binary('Alopecia',handling_strategy='reject_sampling'),
                # Binary('Obesity',handling_strategy='reject_sampling'),
                # Binary('class',handling_strategy='reject_sampling'),
               ]

# Distributions for each field (all set to univariate)
SACardio_distributions = {
    'sbp' : 'univariate',
    'tobacco' : 'univariate',
    'ldl' : 'univariate',
    'adiposity' : 'univariate',
    'famhist' : 'univariate',
    'typea' : 'univariate',
    'obesity' : 'univariate', 
    'alcohol' : 'univariate',
    'age' : 'univariate',
    'chd' : 'univariate', 
    }  