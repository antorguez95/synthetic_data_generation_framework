# Copyright (C) 2022 Antonio Rodriguez
# 
# This file is part of synthetic_data_generation_framework.
# 
# synthetic_data_generation_framework is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# synthetic_data_generation_framework is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with synthetic_data_generation_framework.  If not, see <http://www.gnu.org/licenses/>.

# Dependencies 
import os 
import pandas as pd
import numpy as np

from typing import Callable, Tuple, List 

from  sdg_utils import Positive, Binary

def prepare_PIMA(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List, str] :
    """Read the PIMA dataset from a .csv file and suit it to be processed 
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

def numerical_conversion(data : np.array, features : str, y_col : str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
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
                # Positive('Pregnancies',handling_strategy='reject_sampling'),
                # Positive('Glucose',handling_strategy='reject_sampling'),
                # Positive('BloodPressure',handling_strategy='reject_sampling'),
                # Positive('SkinThickness',handling_strategy='reject_sampling'), 
                # Positive('Insulin',handling_strategy='reject_sampling'),
                # Positive('BMI',handling_strategy='reject_sampling'),
                # Positive('DiabetesPedigreeFunction',handling_strategy='reject_sampling'),
                # Positive('Age',handling_strategy='reject_sampling'),
                # Binary('Outcome',handling_strategy='reject_sampling')
               ]

# Distributions for each field (all set to univariate)
pima_distributions = {
    'Pregnancies' : 'univariate',
    'Glucose' : 'univariate', 
    'BloodPressure' : 'univariate', 
    'SkinThickness' : 'univariate',
    'Insulin' : 'univariate', 
    'BMI' : 'univariate', 
    'DiabetesPedigreeFunction' : 'univariate', 
    'Age' : 'univariate', 
    'Outcome' : 'univariate',
    }

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\PIMA\results"

# Dataset name
dataset_name = 'PIMA'

# Variables needed to handle dictionaries (same as )
# Number of generated data samples 
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Balancing Methods 
balance1 = "ADASYN"
balance2 = "Borderline"

# Augmentation methods
augmen1 = "CTGAN"
augmen2 = "GC"

# Best and worst synthetic generation algorithms combinations
best_worst = ['Borderline + Sep. + GC', 'ADASYN + CTGAN'] # might be wrong

# Best synthetic generation algorithms combination
best_method = 'Borderline + Sep. + GC' # might be wrong

# ML models used and their associated colours
models = ['SVM','RF', 'XGB', 'KNN']
model_colors = ['b','r','k','g']

# Chosen colors for each combinations
ctgan_colors = ["k","r","g","b"]
gc_colors = ["c","m","y","orange"]

# Studied metrics
mets = ["PCD","MMD","KLD"]

# Strings containing combinations of SDG (Synthetic Data Generators) 
comb1 = ("%s + %s") % (balance1, augmen1)
comb2 = ("%s + %s") % (balance1, augmen2)
comb3 = ("%s + %s") % (balance2, augmen1)
comb4 = ("%s + %s") % (balance2, augmen2)
comb5 = ("%s + Sep. + %s") % (balance1, augmen1)
comb6 = ("%s + Sep. + %s") % (balance1, augmen2)
comb7 = ("%s + Sep. + %s") % (balance2, augmen1)
comb8 = ("%s + Sep. + %s") % (balance2, augmen2)
comb9 = "%s" % (augmen1)
comb10 = "Sep. + %s" % (augmen1)

# Split CTGAN and Gaussian Copula methods to plot them separately
ctgan_combinations = [comb1, comb3, comb5, comb7]
gc_combinations = [comb2, comb4, comb6, comb8]

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################