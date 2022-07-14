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

from typing import List, Tuple

def prepare_HeartDiseases(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, 
                    pd.DataFrame, pd.DataFrame, List, str] :
    """Read the Kaggle-HeartDisease dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. It returns tha dataset dataframe and strings associated to 
    it to easy its management.

    Args:
    -----
            dataset_path: path where  dataset is stored. Set by default.
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

def general_conversion(data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Kaggle-HeartDisease database features data types to its original type.
    Categorical variables are set as "object" type. A DataFrame with the original 
    datatypes of this database is returned.

    Args:
    -----
            data: dataset with datatypes not corresponding to the original ones.

    Returns:
    --------
            data: dataframe with the original datatypes 
    """
    data['age'] = data['age'].astype(int)
    data['sex'] = data['sex'].astype(int) #bool
    data['cp'] = data['cp'].astype('object')
    data['trestbps'] = data['trestbps'].astype(int)
    data['chol'] = data['chol'].astype(int)
    data['fbs'] = data['fbs'].astype(int) #bool
    data['restecg'] = data['restecg'].astype('object')
    data['thalach'] = data['thalach'].astype(int)
    data['exang'] = data['exang'].astype(int) #bool
    data['oldpeak'] = data['oldpeak'].astype(float)
    data['slope'] = data['slope'].astype(int)
    data['ca'] = data['ca'].astype(int)
    data['thal'] = data['thal'].astype('object')
    data['target'] = data['target'].astype(int) #bool
    
    return data

def replacement(data : pd.DataFrame) -> pd.DataFrame :
    """This function replaces the numerical values corresponding to categories in 
    the Kaggle-Heart Diseases database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['cp']  = data['cp'].replace([0,1,2,3],['chest_pain_0', 'chest_pain_1', 'chest_pain_2', 'chest_pain_3'])
    data['restecg']  = data['restecg'].replace([0,1,2],['restecg_0', 'restecg_1', 'restecg_2'])
    data['thal']  = data['thal'].replace([0,1,2,3],['thal_0', 'thal_1', 'thal_2', 'thal_3'])

    return data 

def one_hot_enc(data : pd.DataFrame) -> pd.DataFrame :
    """This function performs One-Hot Encoding in the Kaggle-Heart Diseases database.
    Columns full of 0s must be manually added, because a certain value of a feature might not 
    appear in the dataset subset. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # Chest pain 
    cats = ['chest_pain_0', 'chest_pain_1', 'chest_pain_2', 'chest_pain_3']
    data[['cp']] = data[['cp']].astype('category')
    cp = pd.DataFrame(enc.fit_transform(data[['cp']]).toarray())
    cp.columns = enc.categories_
    for name in cats:
        if name not in cp:
            cp[name] = 0
    cp = cp[['chest_pain_0', 'chest_pain_1', 'chest_pain_2', 'chest_pain_3']]
    cp.reset_index(drop=True, inplace=True)
    # ECG resting 
    cats = ['restecg_0', 'restecg_1', 'restecg_2']
    data[['restecg']] = data[['restecg']].astype('category')
    rest = pd.DataFrame(enc.fit_transform(data[['restecg']]).toarray())
    rest.columns = enc.categories_
    for name in cats:
        if name not in rest:
            rest[name] = 0 
    rest = rest[['restecg_0', 'restecg_1', 'restecg_2']]        
    rest.reset_index(drop=True, inplace=True)
    # Thal
    cats = ['thal_0', 'thal_1', 'thal_2', 'thal_3']
    data[['thal']] = data[['thal']].astype('category')
    thal = pd.DataFrame(enc.fit_transform(data[['thal']]).toarray())
    thal.columns = enc.categories_
    for name in cats:
        if name not in thal:
            thal[name] = 0    
    thal = thal[['thal_0', 'thal_1', 'thal_2', 'thal_3']]
    thal.reset_index(drop=True, inplace=True)
    
    # Drop column to add it at the end 
    affected = data[['target']]
    affected.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['cp', 'restecg', 'thal','target'], axis=1)
    data.reset_index(drop=True, inplace=True)
    
    data = pd.concat([data, cp, rest, thal, affected],axis=1)
    
    return data

# Dictionary to specify fields of synthetic data for Kaggle-HeartDisease database
heartDisease_fields = {
    'age' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'sex' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'cp' : {
        'type' : 'categorical'
    },
    'trestbps' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'chol' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'fbs' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'restecg' : {
        'type' : 'categorical'
    },
    'thalach' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'exang' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'oldpeak' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'slope' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'ca' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'thal' : {
        'type' : 'categorical'
    },
    'target' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },   
 }

# Custom variable constraints to generate synthetic data 
constraints = [ 
                #Positive('Edad',handling_strategy='reject_sampling'),
                #Binary('HTAanterior',handling_strategy='reject_sampling'),
                #Positive('TAS',handling_strategy='reject_sampling'), 
                #Positive('TAD',handling_strategy='reject_sampling'),
                #Binary('DM',handling_strategy='reject_sampling'),
                # Positive('Barthel',handling_strategy='reject_sampling'),
                # Positive('Hb',handling_strategy='reject_sampling'),
                # Positive('VCM',handling_strategy='reject_sampling'),
                # Positive('HCM',handling_strategy='reject_sampling'),
                # Positive('Plaquetas',handling_strategy='reject_sampling'),
                # Positive('Leucocitos',handling_strategy='reject_sampling'),
                # Positive('Neutrófilos',handling_strategy='reject_sampling'),
                # Positive('Linfocitos',handling_strategy='reject_sampling'),
                # Positive('Monocitos',handling_strategy='reject_sampling'),
                # Positive('Glucosa',handling_strategy='reject_sampling'),
                # Positive('Creatinina',handling_strategy='reject_sampling'),
                # Positive('FG',handling_strategy='reject_sampling'),
                # Positive('Na',handling_strategy='reject_sampling'),
                # Positive('K',handling_strategy='reject_sampling'),
                # Positive('ALT',handling_strategy='reject_sampling'),
                # Positive('Colesterol',handling_strategy='reject_sampling'),
                # Positive('LDL',handling_strategy='reject_sampling'),
                #Binary('TNeurocog',handling_strategy='reject_sampling'),
                ]

# Distributions for each field (all set to univariate)
heartDisease_distributions = {
    'age' : 'univariate',
    'sex' : 'univariate', 
    'cp' : 'univariate', 
    'trestbps' : 'univariate', 
    'chol' : 'univariate', 
    'fbs' : 'univariate', 
    'restecg' : 'univariate', 
    'thalach' : 'univariate', 
    'exang' : 'univariate', 
    'oldpeak' : 'univariate', 
    'slope' : 'univariate', 
    'ca' : 'univariate',
    'thal' : 'univariate', 
    'target' : 'univariate',    
    }

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\HeartDiseases\results"

# Dataset name 
dataset_name = 'HeartDiseases'

# Variables needed to handle dictionaries (same as )
# Number of generated data samples 
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Balancing Methods 
balance1 = "SMOTE"
balance2 = "NC"

# Augmentation methods
augmen1 = "CTGAN"
augmen2 = "GC"

# Best and worst synthetic data combinations algorithms
best_worst = ['SMOTE + Sep. + GC', 'NC + CTGAN'] # might be wrong

# Best synthetic data combination algorithm 
best_method = 'SMOTE + Sep. + GC' # might be wrong

# ML models used 
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