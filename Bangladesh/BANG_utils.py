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

from sklearn.preprocessing import OneHotEncoder

from typing import Tuple, List 

def cat2num(data: pd.DataFrame) -> pd.DataFrame:
    """This function replaces the categories in the Bangladesh database a number
    in order to be processed by the KNNImputer, since this tool works better with 
    numbers. For more, check sklearn documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. 
    It returns a DataFrame after this replacement. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['Age']  = data['Age'].replace(['Less then 5','Less then 11','Less then 15',
                                        'greater then 15'],[1,2,3,4])
    data['Sex']  = data['Sex'].replace(['Female','Male'],[1,2])
    data['Area of Residence ']  = data['Area of Residence '].replace(['Rural','Suburban',
                                                                        'Urban'],[1,2,3])
    data['HbA1c']  = data['HbA1c'].replace(['Less then 7.5%','Over 7.5%'],[1,2])
    data['Standardized growth-rate in infancy']  = data['Standardized growth-rate in infancy'].replace(['Lowest quartiles','Middle quartiles','Highest quartiles'],[1,2,3])
    data['Standardized birth weight']  = data['Standardized birth weight'].replace(['Lowest quartiles','Middle quartiles','Highest quartiles'],[1,2,3])
    
    return data

def prepare_BANG(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, 
                pd.DataFrame, pd.DataFrame, List, str]:
    """Read the Bangladesh dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This dataset presents lots of categorical and binary data. 
    Sometimes, they are written down differently within different features, or
    binary data is indicated with "yes" or "no". Hence, some transformations must be 
    applied to handle this dataset properly. From "yes" to "1", "No" to "0", etc. 
    Finally, this function converts the categories into numbers to suit the dataset
    for the following steps. This converted DataFrame is returned.

    Args:
    -----
            dataset_path: path where Bangladesh dataset is stored. Set by default.
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

    # Convert "Yes" and "No" values into "1" and "0"
    data.replace(('Yes', 'No'), (1, 0), inplace=True)
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    data.replace(('none', ' none'), ("0","0"), inplace=True)

    # Converting "days", "months" and "years" in a numerical value indicating "days"
    # String by number sustitution
    data['Duration of disease'] = data['Duration of disease'].str.replace('y','*365')
    data['Duration of disease'] = data['Duration of disease'].str.replace('m','*30')
    data['Duration of disease'] = data['Duration of disease'].str.replace('w','*7')
    data['Duration of disease'] = data['Duration of disease'].str.replace('d','*1')

    # Loop to evaluate string expression and substitute it by its correspondant numerical value
    i = 0
    for row in data['Duration of disease']: 
        data['Duration of disease'][i] = eval(data['Duration of disease'][i])
        i = i+1

    # Strings that means 0 replaced by an actual 0
    data.replace(('0'), (0), inplace=True)

    # 'Other disease' transformed into 'Absence' 
    data['Other diease'].loc[data['Other diease'] != 0] = 1

    # "How Taken" feature takes value 'none' or 'Injection', so 'Injection = 1
    data.replace(('Injection'), (1), inplace=True)

    # Substitute 'unknown' by 99999 (considered missing data to further preprocessing)
    data['Standardized birth weight']  = data['Standardized birth weight'].replace(['unkhown'],[99999])
    
    # Converts from category to numerical values 
    data = cat2num(data)
    
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

def numerical_conversion(data : np.array, features : str, y_col : str) -> Tuple(pd.DataFrame, 
                        pd.DataFrame, pd.DataFrame) :
    """Fix all Bangladesh database features data types to its original type after KNNImputer is used,
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
    data['Age'] = data['Age'].astype(int)
    data['Sex'] = data['Sex'].astype(int)
    data['Area of Residence '] = data['Area of Residence '].astype(int)
    data['HbA1c'] = data['HbA1c'].astype(int)
    data['Height'] = data['Height'].astype(float)
    data['Weight'] = data['Weight'].astype(int)
    data['BMI'] = data['BMI'].astype(float)
    data['Duration of disease'] = data['Duration of disease'].astype(float)
    data['Other diease'] = data['Other diease'].astype(int)
    data['Adequate Nutrition '] = data['Adequate Nutrition '].astype(int)
    data['Education of Mother'] = data['Education of Mother'].astype(int)
    data['Standardized growth-rate in infancy'] = data['Standardized growth-rate in infancy'].astype(int)
    data['Standardized birth weight'] = data['Standardized birth weight'].astype(int)
    data['Autoantibodies'] = data['Autoantibodies'].astype(np.int32)
    data['Impaired glucose metabolism '] = data['Impaired glucose metabolism '].astype(int)
    data['Insulin taken'] = data['Insulin taken'].astype(int)
    data['How Taken'] = data['How Taken'].astype(int)
    data['Family History affected in Type 1 Diabetes'] = data['Family History affected in Type 1 Diabetes'].astype(int)
    data['Family History affected in Type 2 Diabetes'] = data['Family History affected in Type 2 Diabetes'].astype(int)
    data['Hypoglycemis'] = data['Hypoglycemis'].astype(int)
    data['pancreatic disease affected in child '] = data['pancreatic disease affected in child '].astype(int)
    data['Affected'] = data['Affected'].astype(int)

    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def general_conversion (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Blangladesh database features data types to its original type.
    Categorical variables are set as "object" type. Binary ones as "bool".
    A DataFrame with the original datatypes of this database is returned.

    Args:
    -----
            data: dataset with datatypes not corresponding to the original ones.

    Returns:
    --------
            data: dataframe with the original datatypes 
    """
    data['Age'] = data['Age'].astype('object')
    data['Sex'] = data['Sex'].astype('object')
    data['Area of Residence '] = data['Area of Residence '].astype('object')
    data['HbA1c'] = data['HbA1c'].astype('object')
    data['Height'] = data['Height'].astype(float)
    data['Weight'] = data['Weight'].astype(int)
    data['BMI'] = data['BMI'].astype(float)
    data['Duration of disease'] = data['Duration of disease'].astype(float)
    data['Other diease'] = data['Other diease'].astype(int)
    data['Adequate Nutrition '] = data['Adequate Nutrition '].astype(int)
    data['Education of Mother'] = data['Education of Mother'].astype(int)
    data['Standardized growth-rate in infancy'] = data['Standardized growth-rate in infancy'].astype('object')
    data['Standardized birth weight'] = data['Standardized birth weight'].astype('object')
    data['Autoantibodies'] = data['Autoantibodies'].astype(int)
    data['Impaired glucose metabolism '] = data['Impaired glucose metabolism '].astype(int)
    data['Insulin taken'] = data['Insulin taken'].astype(int)
    data['How Taken'] = data['How Taken'].astype(int)
    data['Family History affected in Type 1 Diabetes'] = data['Family History affected in Type 1 Diabetes'].astype(int)
    data['Family History affected in Type 2 Diabetes'] = data['Family History affected in Type 2 Diabetes'].astype(int)
    data['Hypoglycemis'] = data['Hypoglycemis'].astype(int)
    data['pancreatic disease affected in child '] = data['pancreatic disease affected in child '].astype(int)
    data['Affected'] = data['Affected'].astype(int)
    
    return data

def num2cat(data : pd.DataFrame) -> pd.DataFrame :
    """This function replaces the numerical values corresponding to categories in 
    the Bangladesh database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['Age']  = data['Age'].replace([1,2,3,4],['Less then 5','Less then 11','Less then 15',
                                        'greater then 15'])
    data['Sex']  = data['Sex'].replace([1,2],['Female','Male'])
    data['Area of Residence ']  = data['Area of Residence '].replace([1,2,3],['Rural','Suburban',
                                                                        'Urban'])
    data['HbA1c']  = data['HbA1c'].replace([1,2],['Less then 7.5%','Over 7.5%'])
    data['Standardized growth-rate in infancy']  = data['Standardized growth-rate in infancy'].replace([1,2,3],['Lowest quartiles','Middle quartiles','Highest quartiles'])
    data['Standardized birth weight']  = data['Standardized birth weight'].replace([1,2,3],['Lowest quartiles','Middle quartiles','Highest quartiles'])
    
    return data 

def one_hot_enc(data) -> pd.DataFrame :
    """This function performs One-Hot Encoding in the Bangladesh database. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # Age
    age = pd.DataFrame(enc.fit_transform(data[['Age']]).toarray())
    age.columns = ["Less then 11", "Less then 15", "Less then 5", "greater then 15"]
    # Sex
    sex = pd.DataFrame(enc.fit_transform(data[['Sex']]).toarray())
    sex.columns = ["Female", "Male"]
    # Area of Residence 
    residence = pd.DataFrame(enc.fit_transform(data[['Area of Residence ']]).toarray())
    residence.columns = ["Rural", "Suburban", "Urban"]
    # HbA1c
    hb = pd.DataFrame(enc.fit_transform(data[['HbA1c']]).toarray())
    hb.columns = ["Less then 7.5%", "Over 7.5%"]
    # Standardized growth-rate infancy 
    growth = pd.DataFrame(enc.fit_transform(data[['Standardized growth-rate in infancy']]).toarray())
    growth.columns = ["Growth Highest quartiles", "Growth Lowest quartiles", "Growth Middle quartiles"]
    # Standardized birth weight 
    weight = pd.DataFrame(enc.fit_transform(data[['Standardized birth weight']]).toarray())
    weight.columns = ["Weight Highest quartiles", "Weight Lowest quartiles", "Weight Middle quartiles"]
    # Drop column to add it at the end 
    affected = data['Affected']
    # Drop original categorical columns
    data = data.drop(['Age', 'Sex', 'HbA1c','Area of Residence ', 'Other diease','Standardized growth-rate in infancy', 'Standardized birth weight', 'Affected' ], axis=1)
    # Joint one-hot encoding columns 
    data = data.join([age, sex, residence, hb, growth, weight, affected])
    
    return data

# Dictionary to specify fields of synthetic data for Bangladesh database
bang_fields = {
    'Age' : {
        'type' : 'categorical'
    },
    'Sex' : {
        'type' : 'categorical'
    },  
    'Area of Residence ' : {
        'type' : 'categorical'
    },    
    'HbA1c' : {
        'type' : 'categorical'
    },    
    'Height' : {
        'type' : 'numerical',
        'subtype' : 'float',
    },
    'Weight' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'BMI' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Duration of disease' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Other diease' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },        
    'Adequate Nutrition ' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }, 
    'Education of Mother' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Standardized growth-rate in infancy' : {
        'type' : 'categorical'
    },    
    'Standardized birth weight' : {
        'type' : 'categorical'
    },          
    'Autoantibodies' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Impaired glucose metabolism ' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }, 
    'Insulin taken' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'How Taken' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },    
    'Family History affected in Type 1 Diabetes' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Family History affected in Type 2 Diabetes' : {
        'type' : 'numerical',
        'subtype' : 'integer'      
   },
    'Hypoglycemis' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'pancreatic disease affected in child ' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Affected' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }           
 }

# Custom variable constraints to generate synthetic data 
constraints =constraints = [ 
                # Positive('Age',handling_strategy='reject_sampling'),
                # Positive('Sex',handling_strategy='reject_sampling'),
                # Positive('Area of residence',handling_strategy='reject_sampling'),
                # Positive('HbA1c',handling_strategy='reject_sampling'),
                # Positive('Height',handling_strategy='reject_sampling'),
                # Positive('Weight',handling_strategy='reject_sampling'),
                # Positive('BMI',handling_strategy='reject_sampling'),
                # Positive('Duration of disease',handling_strategy='reject_sampling'),               
                # Binary('Other diease',handling_strategy='reject_sampling'),                                
                # Binary('Adequate Nutrition ',handling_strategy='reject_sampling'),
                # Binary('Education of Mother',handling_strategy='reject_sampling'), 
                # Positive('Duration of disease',handling_strategy='reject_sampling'),  
                # Positive('Standardized growth-rate in infancy',handling_strategy='reject_sampling'),  
                # Positive('Standardized birth weight',handling_strategy='reject_sampling'), 
                # Binary('Autoantibodies',handling_strategy='reject_sampling'),
                # Binary('Impaired glucose metabolism ',handling_strategy='reject_sampling'),
                # Binary('Insulin taken',handling_strategy='reject_sampling'),
                # Binary('How Taken',handling_strategy='reject_sampling'),
                # Binary('Family History affected in Type 1 Diabetes',handling_strategy='reject_sampling'),
                # Binary('Family History affected in Type 2 Diabetes',handling_strategy='reject_sampling'),
                # Binary('Hypoglycemis',handling_strategy='reject_sampling'),
                # Binary('pancreatic disease affected in child ',handling_strategy='reject_sampling'),
                # Binary('Affected',handling_strategy='reject_sampling')
               ]

# Distributions for each field (all set to univariate)
bang_distributions = {
    'Age' : 'univariate',
    'Sex' : 'univariate',
    'Area of Residence ' : 'univariate',
    'HbA1c' : 'univariate',
    'Height' : 'univariate',
    'Weight' : 'univariate',
    'BMI' : 'univariate', 
    'Duration of disease' : 'univariate',
    'Other diease' : 'univariate',
    'Adequate Nutrition ' : 'univariate', 
    'Education of Mother' : 'univariate', 
    'Standardized growth-rate in infancy' : 'univariate',
    'Standardized birth weight'
    'Autoantibodies' : 'univariate',
    'Impaired glucose metabolism ' : 'univariate',
    'Insulin taken' : 'univariate',
    'How taken' : 'univariate',
    'Family History affected in Type 1 Diabetes' : 'univariate',
    'Family History affected in Type 2 Diabetes' : 'univariate',
    'Hypoglycemis' : 'univariate',
    'pancreatic disease affected in child ' : 'univariate',
    'Affected' : 'univariate', 
    } 

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\Bangladesh\results"

# Dataset name 
dataset_name = 'BANG'

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
best_worst = ['Borderline + Sep. + GC', 'ADASYN + CTGAN'] #might be wrong 

# Best synthetic generation algorithms combinations
best_method = 'Borderline + Sep. + GC' #might be wrong 

# ML models used and associated colours 
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