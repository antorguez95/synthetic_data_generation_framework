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
    """This function replaces the categories in the Chronic Kidney Disease database by a number
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
    
    data['rbc']  = data['rbc'].replace(['normal','abnormal'],[0,1])
    data['pc']  = data['pc'].replace(['normal','abnormal'],[0,1])
    data['appet']  = data['appet'].replace(['good','poor'],[0,1])

    # Even this feature has float numbers, they represent categories
    data['sg'] = data['sg'].replace([1.005, 1.010, 1.015, 1.025],[0, 1, 2, 3])

    return data

def prepare_KidneyChronic(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Kidney Chronic disease dataset from a .csv file and suit it to be processed 
    from .arff format to as a pd.DataFrame. 
    
    This dataset presents lots of categorical and binary data. 
    Sometimes, binary data is indicated with a "yes" or a"no". Hence, some transformations
    must be applied to handle this dataset properly. From "yes" to "1", "No" to "0", etc. 
    Finally, this function converts the categories into numbers to suit the dataset
    for the following steps. This converted DataFrame is returned. 

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
    data = pd.read_csv(filename)

    # Convert "Yes" and "No" (and equivalent) values into "1" and "0" 
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    data.replace(('ckd', 'notckd'), (1,0), inplace=True)
    data.replace(('present', 'notpresent'), (1,0), inplace=True)
    data.replace(('ckd\t'), (1), inplace=True)
    
    # Correcting other erratas
    data.replace(('\t?'), ('nan'), inplace=True)
    data.replace(('\tno', '\tyes'), (0, 1), inplace=True)
    data.replace((' yes'), (1), inplace=True)

    # Replace nan character by np.nan
    data.replace(('nan'), (np.nan), inplace=True)
    data.replace(['nan'],[np.nan])
    
    # Drop ID column 
    data = data.drop(['id'], axis=1)

    # From categories to number to be handled by the balancing algorithms 
    data = cat2num(data)

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

def numerical_conversion(data : np.array, features : str, y_col : str):
    """Fix all Kidney Chronic Disease database features data types to its original 
    type after KNNImputer is used, since this functions returns only a floating points 
    ndarray. For more, check sklearn documentation of this function at
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
    data['age'] = data['age'].astype(int)
    data['bp'] = data['bp'].astype(int)
    data['sg'] = data['sg'].astype(int)
    data['al'] = data['al'].astype(int)
    data['su'] = data['su'].astype(int)
    data['rbc'] = data['rbc'].astype(int)
    data['pc'] = data['pc'].astype(int)
    data['pcc'] = data['pcc'].astype(int) #bool
    data['ba'] = data['ba'].astype(int) #bool
    data['bgr'] = data['bgr'].astype(int) 
    data['bu'] = data['bu'].astype(float)
    data['sc'] = data['sc'].astype(float)
    data['sod'] = data['sod'].astype(float)
    data['pot'] = data['pot'].astype(float)
    data['hemo'] = data['hemo'].astype(float)
    data['pcv'] = data['pcv'].astype(int)
    data['wc'] = data['wc'].astype(int)
    data['rc'] = data['rc'].astype(float)
    data['htn'] = data['htn'].astype(int) #bool
    data['dm'] = data['dm'].astype(int) #bool
    data['htn'] = data['htn'].astype(int) #bool
    data['appet'] = data['appet'].astype(int)
    data['pe'] = data['pe'].astype(int) #bool
    data['ane'] = data['ane'].astype(int) #bool
    data['classification'] = data['classification'].astype(int)

    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def general_conversion (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Kidney Chronic Disease database features data types to its original type.
    Categorical variables are set as "object" type. Binary ones as integer since SDV library
    Gaussian Copulas have some issues yet modelling binary variables.
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
    data['age'] = data['age'].astype(int)
    data['bp'] = data['bp'].astype(int)
    data['sg'] = data['sg'].astype('object')
    data['al'] = data['al'].astype('object')
    data['su'] = data['su'].astype('object')
    data['rbc'] = data['rbc'].astype('object')
    data['pc'] = data['pc'].astype('object')
    data['pcc'] = data['pcc'].astype(int) #bool
    data['ba'] = data['ba'].astype(int) #bool
    data['bgr'] = data['bgr'].astype(int) 
    data['bu'] = data['bu'].astype(float)
    data['sc'] = data['sc'].astype(float)
    data['sod'] = data['sod'].astype(float)
    data['pot'] = data['pot'].astype(float)
    data['hemo'] = data['hemo'].astype(float)
    data['pcv'] = data['pcv'].astype(int)
    data['wc'] = data['wc'].astype(int)
    data['rc'] = data['rc'].astype(float)
    data['htn'] = data['htn'].astype(int) #bool
    data['dm'] = data['dm'].astype(int) #bool
    data['cad'] = data['cad'].astype(int) #bool
    data['appet'] = data['appet'].astype('object')
    data['pe'] = data['pe'].astype(int) #bool
    data['ane'] = data['ane'].astype(int) #bool
    data['classification'] = data['classification'].astype(int)
    
    return data

def num2cat(data : pd.DataFrame):
    """This function replaces the numerical values corresponding to categories in 
    the KidneyChronic database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['rbc']  = data['rbc'].replace([0,1],['rbc_normal','rbc_abnormal'])
    data['pc']  = data['pc'].replace([0,1],['pc_normal','pc_abnormal'])
    data['appet']  = data['appet'].replace([0,1],['appet_good','appet_poor'])

    # Numerical values that represent categories
    data['sg'] = data['sg'].replace([0, 1, 2, 3], ['sg_1.005', 'sg_1.010', 'sg_1.015', 'sg_1.025'])
    data['al'] = data['al'].replace([0, 1, 2, 3, 4, 5], ['al_0', 'al_1', 'al_2', 'al_3', 'al_4', 'al_5'])
    data['su'] = data['su'].replace([0, 1, 2, 3, 4, 5], ['su_0', 'su_1', 'su_2', 'su_3', 'su_4', 'su_5'])
 
    return data 

def one_hot_enc(data):
    """This function performs One-Hot Encoding in the KidneyChronic database. Sometimes 
    columns full of 0s must be manually added, because a certain value of a certain feature
    might not appear in the dataset subset.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')

    # sg
    cats = ['sg_1.005', 'sg_1.010', 'sg_1.015', 'sg_1.025']
    data[['sg']] = data[['sg']].astype('category')
    sg = pd.DataFrame(enc.fit_transform(data[['sg']]).toarray())
    sg.columns = enc.categories_
    for name in cats:
        if name not in sg:
            sg[name] = 0
    sg.reset_index(drop=True, inplace=True)
    # al
    cats = ['al_0', 'al_1', 'al_2', 'al_3', 'al_4', 'al_5']
    data[['al']] = data[['al']].astype('category')
    al = pd.DataFrame(enc.fit_transform(data[['al']]).toarray())
    al.columns = enc.categories_
    for name in cats:
        if name not in al:
            al[name] = 0    
    al.reset_index(drop=True, inplace=True)
    # su
    cats = ['su_0', 'su_1', 'su_2', 'su_3', 'su_4', 'su_5']
    data[['su']] = data[['su']].astype('category')
    su = pd.DataFrame(enc.fit_transform(data[['su']]).toarray())
    su.columns = enc.categories_
    for name in cats:
        if name not in su:
            su[name] = 0    
    su.reset_index(drop=True, inplace=True)
    # rbc
    cats = ['rbc_normal','rbc_abnormal']
    data[['rbc']] = data[['rbc']].astype('category')
    rbc = pd.DataFrame(enc.fit_transform(data[['rbc']]).toarray())
    rbc.columns = enc.categories_
    for name in cats:
        if name not in rbc:
            rbc[name] = 0   
    rbc.reset_index(drop=True, inplace=True)
    # pc
    cats = ['pc_normal','pc_abnormal']
    data[['pc']] = data[['pc']].astype('category')
    pc = pd.DataFrame(enc.fit_transform(data[['pc']]).toarray())
    pc.columns = enc.categories_
    for name in cats:
        if name not in pc:
            pc[name] = 0     
    pc.reset_index(drop=True, inplace=True)
    # appet
    cats = ['appet_good','appet_poor']
    data[['appet']] = data[['appet']].astype('category')
    app = pd.DataFrame(enc.fit_transform(data[['appet']]).toarray())
    app.columns = enc.categories_
    for name in cats:
        if name not in app:
            app[name] = 0     
    app.reset_index(drop=True, inplace=True)

    # Drop target variable column to add it at the end 
    clas = data[['classification']]
    clas.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['classification'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Drop the original categorical column 
    data = data.drop(['sg','al','su','rbc','pc','appet'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Joint one-hot encoding columns 
    data = data.join([sg, al, su, rbc, pc, app, clas])
    
    return data

# Dictionary to specify fields of synthetic data for Alzheimer-Balea database
kidneyChronic_fields = {
    'age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'bp' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },  
    'sg' : {
        'type' : 'categorical'
    }, 
    'al' : {
        'type' : 'categorical'
    } ,
    'su' : {
        'type' : 'categorical',
    } ,
    'rbc' : {
        'type' : 'categorical'
    } ,
    'pc' : {
        'type' : 'categorical'
    } ,
    'pcc' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'ba' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'bgr' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'bu' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'sc' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'sod' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'pot' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'hemo' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'pcv' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'wc' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'rc' : {
        'type' : 'numerical',
        'subtype' : 'float'
    } ,
    'htn' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'dm' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'htn' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'appet' : {
        'type' : 'categorical'
    } ,
    'pe' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'cad' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    } ,
    'classification' : {
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
kidneyChronic_distributions = {
    'age' : 'univariate',
    'bp' : 'univariate',
    'sg' : 'univariate',
    'al' : 'univariate',
    'su' : 'univariate',
    'rbc' : 'univariate',
    'pc' : 'univariate', 
    'pcc' : 'univariate',
    'ba' : 'univariate',
    'bgr' : 'univariate', 
    'bu' : 'univariate', 
    'sc' : 'univariate',
    'sod' : 'univariate',
    'pot' : 'univariate',
    'hemo' : 'univariate',
    'pcv' : 'univariate',
    'wc' : 'univariate',
    'rc' : 'univariate',
    'htn' : 'univariate',
    'dm' : 'univariate',
    'cad' : 'univariate',
    'appet' : 'univariate',
    'pe' : 'univariate',
    'ane' : 'univariate',
    'classification' : 'univariate',
    }  

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\KidneyChronicDisease\results"

dataset_name = 'KidneyChronic'

# Variables needed to handle dictionaries (same as )
# Number of generated data samples 
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Balancing Methods 
balance1 = "ADASYN"
balance2 = "Borderline"

# Augmentation methods
augmen1 = "CTGAN"
augmen2 = "GC"

best_worst = ['Borderline + Sep. + GC', 'ADASYN + CTGAN'] 

best_method = 'Borderline + Sep. + GC'

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