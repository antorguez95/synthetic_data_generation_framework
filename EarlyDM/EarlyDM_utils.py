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

from  sdg_utils import Positive, Binary 

from typing import Tuple

from sklearn.preprocessing import OneHotEncoder

def cat2num(data: pd.DataFrame):
    """This function replaces the categories in the Early-DM database by a number
    in order to be processed by the balancing algorithms, since this tool works better with 
    numbers. For more information, check https://imbalanced-learn.org/stable/. 
    It returns a DataFrame after this replacement. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['Gender']  = data['Gender'].replace(['Male','Female'],[0,1])
    
    return data

def prepare_EarlyDM(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Early-DM dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This dataset presents lots of categorical and binary data. 
    Sometimes, binary data is indicated with a "yes" or a"no". Hence, some transformations
    must be applied to handle this dataset properly. From "yes" to "1", "No" to "0", etc. 
    Finally, this function converts the categories into numbers to suit the dataset
    for the following steps. This converted DataFrame is returned. 

    Args:
    -----
            dataset_path: path where dataset is stored. Set by default.
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

    # Open the .csv file and convert it into DataFrame
    data = pd.read_csv(filename)

    # Store column names 
    cols_names = data.columns

    # Replace 'Yes' and 'Positive', and 'No' and 'Negative' by 1 and 0 respectively.
    data.replace(('Yes', 'No'), (1, 0), inplace=True)
    data.replace(('Positive', 'Negative'), (1, 0), inplace=True)

    # Store features' and target variable's names 
    cols_names_prev = data.columns
    y_tag = cols_names_prev[len(cols_names_prev)-1]
    cols_names = cols_names_prev[0:cols_names_prev.size]

    # From categories to number to be handled by the balancing algorithms 
    data = cat2num(data)

    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

def general_conversion (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Blangladesh database features data types to its original type.
    Categorical variables are set as "object" type. Binary ones as "bool".
    A DataFrame with the original datatypes of this database is returned.

    Args:
    -----
            data: dataset with datatypes not corresponding to the original ones.
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe with the original datatypes 
    """
    data['Age'] = data['Age'].astype(int)
    data['Gender'] = data['Gender'].astype('object')
    data['Polyuria'] = data['Polyuria'].astype(int)
    data['Polydipsia'] = data['Polydipsia'].astype(int)
    data['Polyphagia'] = data['Polyphagia'].astype(int)
    data['sudden weight loss'] = data['sudden weight loss'].astype(int)
    data['weakness'] = data['weakness'].astype(int)
    data['Genital thrush'] = data['Genital thrush'].astype(int)
    data['visual blurring'] = data['visual blurring'].astype(int)
    data['Itching'] = data['Itching'].astype(int)
    data['Irritability'] = data['Irritability'].astype(int)
    data['delayed healing'] = data['delayed healing'].astype(int)
    data['partial paresis'] = data['partial paresis'].astype(int)
    data['muscle stiffness'] = data['muscle stiffness'].astype(int)
    data['Alopecia'] = data['Alopecia'].astype(int)
    data['Obesity'] = data['Obesity'].astype(int)
    data['class'] = data['class'].astype(int)
    
    return data

def num2cat(data : pd.DataFrame):
    """This function replaces the numerical values corresponding to categories in 
    the EarlyDM database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['Gender']  = data['Gender'].replace([0,1],['Male','Female'])
 
    return data 

def one_hot_enc(data):
    """This function performs One-Hot Encoding in the EarlyDM database. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # Gender
    data[['Gender']] = data[['Gender']].astype('category')
    gen = pd.DataFrame(enc.fit_transform(data[['Gender']]).toarray())
    gen.columns = enc.categories_
    gen.reset_index(drop=True, inplace=True)

    # Drop target variable column to add it at the end 
    clas = data[['class']]
    clas.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['class'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Drop the original categorical column 
    data = data.drop(['Gender'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Joint one-hot encoding columns 
    data = data.join([gen, clas])
    
    return data

# Dictionary to specify fields of synthetic data for Alzheimer-Balea database
# earlyDM_fields = {
#     'Age' : {
#         'type' : 'numerical',
#         'subtype' : 'integer'
#     },
#     'Gender' : {
#         'type' : 'categorical'
#     },  
#     'Polyuria' : {
#         'type' : 'boolean'
#     }, 
#     'Polydipsia' : {
#         'type' : 'boolean'
#     } ,
#     'Polyphagia' : {
#         'type' : 'boolean'
#     } ,
#     'sudden weight loss' : {
#         'type' : 'boolean'
#     } ,
#     'weakness' : {
#         'type' : 'boolean'
#     } ,
#     'Genital thrush' : {
#         'type' : 'boolean'
#     } ,
#     'visual blurring' : {
#         'type' : 'boolean'
#     } ,
#     'Itching' : {
#         'type' : 'boolean'
#     } ,
#     'Irritability' : {
#         'type' : 'boolean'
#     } ,
#     'delayed healing' : {
#         'type' : 'boolean'
#     } ,
#     'partial paresis' : {
#         'type' : 'boolean'
#     } ,
#     'muscle stiffness' : {
#         'type' : 'boolean'
#     } ,
#     'Alopecia' : {
#         'type' : 'boolean'
#     } ,
#     'Obesity' : {
#         'type' : 'boolean'
#     } ,
#     'class' : {
#         'type' : 'boolean'
#     }           
#  }

earlyDM_fields = {
    'Age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Gender' : {
        'type' : 'categorical'
    },  
    'Polyuria' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }, 
    'Polydipsia' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Polyphagia' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'sudden weight loss' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'weakness' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Genital thrush' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'visual blurring' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Itching' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Irritability' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'delayed healing' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'partial paresis' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'muscle stiffness' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Alopecia' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'Obesity' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'class' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }           
 }

# Custom variable constraints to generate synthetic data 
constraints =constraints = [ 
                Positive('Age',handling_strategy='reject_sampling'),              
                Binary('Polyuria',handling_strategy='reject_sampling'),                                
                Binary('Polyphagia',handling_strategy='reject_sampling'),
                Binary('Polydipsia',handling_strategy='reject_sampling'), 
                Binary('sudden weight loss',handling_strategy='reject_sampling'),
                Binary('weakness',handling_strategy='reject_sampling'),
                Binary('Genital thrush',handling_strategy='reject_sampling'),
                Binary('visual blurring',handling_strategy='reject_sampling'),
                Binary('Irritability',handling_strategy='reject_sampling'),
                Binary('Itching',handling_strategy='reject_sampling'),
                Binary('delayed healing',handling_strategy='reject_sampling'),
                Binary('partial paresis',handling_strategy='reject_sampling'),
                Binary('muscle stiffness',handling_strategy='reject_sampling'),
                Binary('Alopecia',handling_strategy='reject_sampling'),
                Binary('Obesity',handling_strategy='reject_sampling'),
                Binary('class',handling_strategy='reject_sampling'),
               ]

# Distributions for each field (all set to univariate)
earlyDM_distributions = {
    'Age' : 'univariate',
    'Gender' : 'univariate',
    'Polyuria' : 'univariate',
    'Polydipsia' : 'univariate',
    'Polyphagia' : 'univariate',
    'sudden weight loss' : 'univariate',
    'weakness' : 'univariate', 
    'Genital thrush' : 'univariate',
    'visual blurring' : 'univariate',
    'Itching' : 'univariate', 
    'Irritability' : 'univariate', 
    'delayed healing' : 'univariate',
    'partial paresis' : 'univariate',
    'Alopecia' : 'univariate',
    'Obesity' : 'univariate',
    'class' : 'univariate',
    }  

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\EarlyDM\results"

dataset_name = 'Early-DM'

# Variables needed to handle dictionaries (same as )
# Number of generated data samples 
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Balancing Methods 
balance1 = "NC"
balance2 = "SVM-SMOTE"

# Augmentation methods
augmen1 = "CTGAN"
augmen2 = "GC"

best_worst = ['NC + Sep. + GC', 'SVM-SMOTE + CTGAN'] 

best_method = 'NC + Sep. + GC'

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