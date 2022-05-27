# Dependencies 
import os 
import pandas as pd
import numpy as np

from  sdg_utils import Positive, Binary 

from typing import Tuple

from sklearn.preprocessing import OneHotEncoder

def prepare_Framingham(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Framingham dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This converted DataFrame is returned. 

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

    # Replace nan values by np.nan and 
    data.replace(('nan'), (np.nan), inplace=True)

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

def numerical_conversion(data : np.array, features : str, y_col : str):
    """Fix all Framingham database features data types to its original type after KNNImputer is used,
    since this functions returns only a floating points ndarray. For more, check sklearn 
    documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. After 
    fixing datatypes, an ndarray to pd.DataFrame conversion is performed. Notice that this
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
    
    # Fixing necessary datatypes to int (including categorical variables)
    data['male'] = data['male'].astype(int)
    data['age'] = data['age'].astype(int)
    data['education'] = data['education'].astype(int)
    data['currentSmoker'] = data['currentSmoker'].astype(int)
    data['cigsPerDay'] = data['cigsPerDay'].astype(int)
    data['BPMeds'] = data['BPMeds'].astype(int)
    data['prevalentStroke'] = data['prevalentStroke'].astype(int)
    data['prevalentHyp'] = data['prevalentHyp'].astype(int)
    data['diabetes'] = data['diabetes'].astype(int)
    data['totChol'] = data['totChol'].astype(int)
    data['sysBP'] = data['sysBP'].astype(float)
    data['diaBP'] = data['diaBP'].astype(float)
    data['BMI'] = data['BMI'].astype(float)
    data['heartRate'] = data['heartRate'].astype(int)
    data['glucose'] = data['glucose'].astype(int)
    data['TenYearCHD'] = data['TenYearCHD'].astype(int)
    
    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def general_conversion (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Framingham database features data types to its original type.
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
    data['male'] = data['male'].astype(int)
    data['age'] = data['age'].astype(int)
    data['education'] = data['education'].astype('object')
    data['currentSmoker'] = data['currentSmoker'].astype(int)
    data['cigsPerDay'] = data['cigsPerDay'].astype(int)
    data['BPMeds'] = data['BPMeds'].astype(int)
    data['prevalentStroke'] = data['prevalentStroke'].astype(int)
    data['prevalentHyp'] = data['prevalentHyp'].astype(int)
    data['diabetes'] = data['diabetes'].astype(int)
    data['totChol'] = data['totChol'].astype(int)
    data['sysBP'] = data['sysBP'].astype(float)
    data['diaBP'] = data['diaBP'].astype(float)
    data['BMI'] = data['BMI'].astype(float)
    data['heartRate'] = data['heartRate'].astype(int)
    data['glucose'] = data['glucose'].astype(int)
    data['TenYearCHD'] = data['TenYearCHD'].astype(int)
    
    return data

def num2cat(data : pd.DataFrame):
    """This function replaces the numerical values corresponding to categories in 
    the Framingham database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['education']  = data['education'].replace([1,2,3,4],['edu1','edu2','edu3','edu4'])
 
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
    # education
    data[['education']] = data[['education']].astype('category')
    edu = pd.DataFrame(enc.fit_transform(data[['education']]).toarray())
    edu.columns = enc.categories_
    edu.reset_index(drop=True, inplace=True)

    # Drop target variable column to add it at the end 
    clas = data[['TenYearCHD']]
    clas.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['TenYearCHD'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Drop the original categorical column 
    data = data.drop(['education'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Joint one-hot encoding columns 
    data = data.join([edu, clas])
    
    return data

# Dictionary to specify fields of synthetic data for Alzheimer-Balea database
framingham_fields = {
    'male' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },  
    'education' : {
        'type' : 'categorical'
    }, 
    'currentSmoker' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'cigsPerDay' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'BPMeds' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'prevalentStroke' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'prevalentHyp' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'diabetes' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'totChol' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'sysBP' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'diaBP' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'BMI' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'heartRate' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'glucose' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'TenYearCHD' : {
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
framingham_distributions = {
    'male' : 'univariate',
    'age' : 'univariate',
    'education' : 'univariate',
    'currentSmoker' : 'univariate',
    'cigsPerDay' : 'univariate',
    'BPMeds' : 'univariate',
    'prevalentStroke' : 'univariate',
    'prevalentHyp' : 'univariate',
    'diabetes' : 'univariate',
    'totChol' : 'univariate',
    'sysBP' : 'univariate', 
    'diaBP' : 'univariate',
    'BMI' : 'univariate',
    'heartRate' : 'univariate', 
    'diaBP' : 'univariate', 
    'BMI' : 'univariate',
    'heartRate' : 'univariate',
    'glucose' : 'univariate',
    'TenYearCHD' : 'univariate',
    }  