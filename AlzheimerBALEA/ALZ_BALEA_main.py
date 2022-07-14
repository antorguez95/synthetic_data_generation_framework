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
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import time 

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.svm import SVC

import sys 
sys.path.append('..')

from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, KMeansSMOTE, SVMSMOTE, BorderlineSMOTE

from ALZ_BALEA_utils import *

from exploratory_data_analysys import *

from sdg_utils import * 

from sdv import Metadata
from sdv.tabular import GaussianCopula

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from train_utils import *

from model_evaluation import *

import pickle

#########################################################################################################
#########################################################################################################
#########################################################################################################
####################################      ARGUMENTS       ##############################################
# Dataset path
DATASET_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Alzheimer\Balea"

# File name 
filename = "alzheimer_BD.xlsx"

# Dataset name
dataset_name = 'ALZ-BALEA'

# Path to store obtained reusults
STORE_PATH = r".\results"

# Flag: set to True if Balancng evaluation has been done. To True if has been not.
BALANCING_CHECKED = False 

# Balancing methods list. For some databases, balancing algorithm may give an error, 
# or has no sense to test is (i.e., SMOTE-NC with a dataset without categorical variables).
# In those cases, replace that value with "None". Order must be always: 
# ["ADASYN", "SMOTE", "SMOTENC", "KMeansSMOTE", "SVMSMOTE", "BorderlineSMOTE"].
# Check "balancing_eval()" function for more. 
bal_methods = [None, "SMOTE", "SMOTENC", None, "SVMSMOTE", "BorderlineSMOTE"]

# Best balancing algorithms for the given dataset. After balancing evaluation is done, 
# fill these values properly. This strings must correspond to the methods that generates
# X_balance1 and X_balance2.
balance1 = "NC"
balance2 = "Borderline"

# Strings that contain the names of the augmentation methods. This is fixed in this framework 
augmen1 = "CTGAN"
augmen2 = "GC"

# Set number of iterations to be done for the balancing evaluation 
bal_iterations = 100

# Number of iterations to evaluate the process of data augmentation 
aug_iterations = 10

# ML Classifiers and their correspondant hyperparameters. If one wants to add any ML classifiers,
# their hyperparameters must be added to. Besides, code in this main file must be added. 'CTRL+F' and
# type 'SVM', and replies that line of code with the new introduced model. 
# SVM 
svm_model = SVC(random_state = 12345, cache_size=200, max_iter = -1, probability=True)
svm_params = {"kernel" : ['rbf', 'linear'],
              "C" : [0.1, 1, 2.5, 5, 10],
              "gamma" : [0.01, 0.1, 1, 10],
              }

# RF
rf_params = {"n_estimators": [20, 50, 100, 200], 
              "max_features": [2,3,5,7],
              }     
rf_model = RandomForestClassifier(random_state = 12345)

# XGB 
xgb_params = {"learning_rate": [0.01, 0.1, 0.5],
              "n_estimators": [20, 50, 100, 200]
              }
xgb_model = GradientBoostingClassifier(random_state = 12345)

# KNN
knn_params = {"n_neighbors": [6,8,10,12,14,16],
              "weights" : ['uniform','distance'],
              }
knn_model = KNeighborsClassifier(algorithm = 'auto', n_jobs = -1)

# Strings to handle ML models and their correspondant colours to be plotted
models = ['SVM','RF', 'XGB', 'KNN']
model_colors = ['b','r','k','g']

# List with the string keys to handle evaluated sizes. Changes here must correspont
# with changes in "sizes_balance1" and "sizes_balance2" variables 
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Studied statistical metrics. If metrics wants to be added, must be added in "sdg_utils.py" file. They should 
# be implemented in the same part of the code as the rest are placed in this main.  
mets = ["PCD","MMD","KLD"]

# Studied classification metrics. To introduce more metrics, modify "model_evaluation.py" file
class_metrics = ['acc', 'auc', 'f1'] 
    
# Chosen colors for each combinations
ctgan_colors = ["k","r","g","b"]
gc_colors = ["c","m","y","orange"]

# Categorical features indexes
cat_feat_idxs = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21]

# Differentiating between categorical and numerical features. The former are 
# one-hot encoded and the latter are standardized
categorical_features = ['Sexo', 'EstCivil', 'ActLaboral', 'FormatConvivencia',
                        'NivelEducativo', 'ActIntelectual', 'RelSociales',
                        'AnPsiquia', 'AntCardiologi', 'AntNeurolog','AntRenal',
                        'AntPulmonar','AntDemencia','Tabaco', 'Alcohol']
numerical_features = ['Edad','TAS','TAD','Barthel','Hb','VCM',
                      'Plaquetas','Leucocitos','Neutrófilos','Linfocitos',
                      'Monocitos','Glucosa','Creatinina','FG','Na','K',
                      'ALT','Colesterol','LDL']

####################################       END OF ARGUMENTS       #######################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

####################################           MAIN BODY          #######################################

# Save working directory to return to it 
wd = os.getcwd()

# Monitoring computational time 
start = time.time()

# Prepare Alzheimer-Balea database to be handled
data, X, Y, feat_names, y_tag = prepare_ALZ_BALEA(dataset_path = DATASET_PATH, filename = filename)

# Return from dataset directory to working directory 
os.chdir(wd)

# Exploratory Data Analysis
#eda(data, X, Y, dataset_name)

# Data partition - Train (80%) + Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=4)
print ('Training set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)

# Concatenate X and Y dataframes 
train_data = pd.concat([X_train, y_train], axis = 1)
validation_data = pd.concat([X_val, y_val], axis = 1)

# KNN-Imputation (training and validation set separately)
# Imputer declaration 
imputer = KNNImputer(missing_values = 99999 , n_neighbors = 5, weights = 'uniform', metric = 'nan_euclidean',
                      copy = 'false') 

# Imputation (training and validation separately)
train_data = imputer.fit_transform(train_data)
validation_data = imputer.fit_transform(validation_data) 

# Conversion from np.array to pd.DataFrame and convert from float to original datatype
train_data, X_train, y_train = numerical_conversion(train_data, feat_names, y_tag)
validation_data, X_val, y_val = numerical_conversion(validation_data, feat_names, y_tag)

# Separate control and cases to duplicate together and separated
controls = train_data.loc[(train_data[y_tag]==0)] 
cases = train_data.loc[(train_data[y_tag]==1)]

# Calculate cases/control ratio
train_ratio = (train_data[y_tag][train_data[y_tag] == 1].value_counts()[1])/(train_data[y_tag][train_data[y_tag] == 0].value_counts()[0])

# Exploratory Data Analysis after KNN Imputation 
# eda(train_data, X_train, y_train, dataset_name, folder = r"./EDA_train")
# eda(validation_data, X_val, y_val, dataset_name, folder = r"./EDA_val")

#%% Data augmentation models: Balancing and Augmentation steps 
# a) Balancing data 
# This piece of code is for estimating which balancing models better work on this database. 
# Flag "CHECKED" should be set to True after evaluation has to skip the balancing evaluation
# part once this step was performed 

# Balancing algorithms evaluation
if BALANCING_CHECKED  == False : 
    print("Balancing algorithms evaluation... %i iterations running" % bal_iterations)
    balancing_eval(dataset_name, X_train, y_train, train_data,
                 feat_names, y_tag, bal_methods, 
                  cat_feat_idxs, filename = "balancing_metrics.csv" , iterations = bal_iterations, store_path = STORE_PATH)

# In case balancing evaluation is not done, change directory
else: 
    os.chdir(STORE_PATH)
                
#%% A) Data Balancing 

# Balancing with chosen method I 
X_balance1, y_balance1 = SMOTENC(categorical_features = cat_feat_idxs,
                                              sampling_strategy = 'minority', 
                                              random_state = None,
                                              k_neighbors = 5,
                                              n_jobs = None).fit_resample(X_train, y_train)
# Balancing with chosen method II   
X_balance2, y_balance2 = BorderlineSMOTE(sampling_strategy = 'minority',
                                                random_state = None,
                                                k_neighbors = 10,
                                                n_jobs = None,
                                                m_neighbors = 10,
                                                kind = 'borderline-1').fit_resample(X_train, y_train)    
    
# Add column Y  to dataframe
# SMOTE-NC 
X_balance1.reset_index(drop=True, inplace=True)
y_balance1.reset_index(drop=True, inplace=True)
data_balance1= pd.concat([X_balance1, y_balance1], axis = 1)

# Borderline-SMOTE
X_balance2.reset_index(drop=True, inplace=True)
y_balance2.reset_index(drop=True, inplace=True)
data_balance2 = pd.concat([X_balance2, y_balance2], axis = 1)

# From datasets, replace HTA and DM 2 and 1, by 1 and 0
train_data['HTAanterior'] = train_data['HTAanterior'].replace([1,2],[1,0])
data_balance1['HTAanterior'] = data_balance1['HTAanterior'].replace([1,2],[1,0])
data_balance2['HTAanterior'] = data_balance2['HTAanterior'].replace([1,2],[1,0])
train_data['DM'] = train_data['DM'].replace([1,2],[1,0])
data_balance1['DM'] = data_balance1['DM'].replace([1,2],[1,0])
data_balance2['DM'] = data_balance2['DM'].replace([1,2],[1,0])

# Data type conversion and control/cases splitting to generate samples separately
# Training set 
train_data = general_conversion(train_data)
controls = train_data.loc[(train_data[y_tag]==0)] 
cases = train_data.loc[(train_data[y_tag]==1)]

# Balance 1
data_balance1 = general_conversion(data_balance1) 
controls_balance1 = data_balance1.loc[(data_balance1[y_tag]==0)] 
cases_balance1 = data_balance1.loc[(data_balance1[y_tag]==1)] 

# Balance 2
data_balance2 = general_conversion(data_balance2)
controls_balance2 = data_balance2.loc[(data_balance2[y_tag]==0)] 
cases_balance2 = data_balance2.loc[(data_balance2[y_tag]==1)]

# Validation (non-splitted)
validation_data = general_conversion(validation_data)

# Defining metadata for Alzheimer-Balea database
metadata = Metadata()
metadata.add_table(
    name = dataset_name,
    data = train_data,
    fields_metadata = alz_fields)

# A) Gaussian Copula model 
gc = GaussianCopula(table_metadata = metadata._metadata['tables'][dataset_name]) 
gc = GaussianCopula(field_types = alz_fields, 
                    #constraints = constraints, 
                    field_distributions = alz_distributions)
 
# B) CTGAN model 
from sdv.tabular import CTGAN
ctgan = CTGAN(field_types = alz_fields,
              constraints=constraints, 
              cuda = True)

# Set conditions for synthetic generation 
cond_positive = {
    y_tag: 1
    }
cond_negative = {
    y_tag: 0
    }

sizes_balance1 = [round(data_balance1.shape[0]/4), round(data_balance1.shape[0]/2), round(data_balance1.shape[0]), 
         round(data_balance1.shape[0]*2), round(data_balance1.shape[0]*4), round(data_balance1.shape[0]*4)+20]
sizes_balance2 = [round(data_balance2.shape[0]/4), round(data_balance2.shape[0]/2), round(data_balance2.shape[0]), 
         round(data_balance2.shape[0]*2), round(data_balance2.shape[0]*4), round(data_balance2.shape[0]*4)+20]
sizes_only_augmen1 = [round(X_train.shape[0]/4), round(X_train.shape[0]/2), round(X_train.shape[0]), 
         round(X_train.shape[0]*2), round(X_train.shape[0]*4), round(data_balance2.shape[0]*4)+20]

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

# Methods put alltogether to create dictionary keys 
sdg_combinations = [comb1, comb2, comb3, comb4, comb5,
           comb6, comb7, comb8, comb9, comb10]

sdg_metrics, class_metrics, hyperparameters = get_eval_dictionaries(sdg_combinations, sizes_keys, class_metrics, aug_iterations)

# List of tuples containing combinations of balancing, augmenting methods and their associated strings
# Non-splitting and no conditions tuples
no_split = [(augmen1, balance1, data_balance1, sizes_balance1),
(augmen1, balance2, data_balance2, sizes_balance2),
(augmen2, balance1, data_balance1, sizes_balance1),
(augmen2, balance2, data_balance2, sizes_balance1)]

# Splitting with no conditions tuples
split = [(augmen1, balance1, controls_balance1, cases_balance1, sizes_balance1),
(augmen1, balance2, controls_balance2, cases_balance2, sizes_balance2),
(augmen2, balance1, controls_balance1, cases_balance1, sizes_balance1),
(augmen2, balance2, controls_balance2, cases_balance2, sizes_balance2)]

# Replacement of numbers by categories to make one-hot encoding aferwards
################################ CLARIFICATIONS ############################
# 'Epilepsia' value in 'AntNeurologi' feature, 'Cancer Pulmonar' value in 'Ant Pulmonar'
# and 'Renal Otras' in 'AntRenal' feature do not appear. These values will be thus
#  deleted. In a greater database, such change might not be necessary 
######################################################################## 

# Performs replacement and one-hot enconding in the training and validation set without synthetic samples
train_data = replacement(train_data)
train_data = one_hot_enc(train_data)
validation_data = replacement(validation_data)
validation_data = one_hot_enc(validation_data)

# Drops unexistant categories in training and validation subsets dataset 
train_data = train_data.drop([('Pulmonar Cancer',)],axis=1)
train_data = train_data.drop([('Neuro Epilepsia',)],axis=1)
train_data = train_data.drop([('Renal Otras',)],axis=1)
cols_names = train_data.columns[0:len(train_data.columns)-1]
X_train = train_data[cols_names]
validation_data = validation_data.drop([('Pulmonar Cancer',)],axis=1)
validation_data = validation_data.drop([('Neuro Epilepsia',)],axis=1)
validation_data = validation_data.drop([('Renal Otras',)],axis=1)
X_val = validation_data.drop(['TNeurocog'], axis=1)
y_val = validation_data[['TNeurocog']]

# Computes training set reference metrics just once to compare it to the different synthetic methods
train_PMFs, train_hist_bases = PMF(X_train)

# Standardization of Training and Validation set to train and validate
X_train_norm, y_train_norm = standardization_cat(X_train, y_train, numerical_features)
X_val_norm, y_val_norm = standardization_cat(X_val, y_val, numerical_features)

# Control/Cases ratio for ADASYN and Borderline 
balance1_ratio = (data_balance1[y_tag][data_balance1[y_tag] == 1].value_counts()[1])/(data_balance1[y_tag][data_balance1[y_tag] == 0].value_counts()[0])
balance2_ratio = (data_balance2[y_tag][data_balance2[y_tag] == 1].value_counts()[1])/(data_balance2[y_tag][data_balance2[y_tag] == 0].value_counts()[0])

# Models' training with the original training sets without synthetic samples 
SVM_model, SVM_train_results, SVM_cv_results  = model_train(svm_model, svm_params, "SVM", "No synthetic", X_train_norm, y_train_norm, cv = 10, scoring = 'f1')
rf_model, rf_train_results, rf_cv_results  = model_train(rf_model, rf_params, "RF", "No synthetic", X_train_norm, y_train_norm, cv = 10, scoring = 'f1')
xgb_model, xgb_train_results, xgb_cv_results  = model_train(xgb_model, xgb_params, "SVM", "No synthetic", X_train_norm, y_train_norm, cv = 10, scoring = 'f1')
knn_model, knn_train_results, knn_cv_results  = model_train(knn_model, knn_params, "SVM", "No synthetic", X_train_norm, y_train_norm, cv = 10, scoring = 'f1')

# Models' evaluation with the original training sets without synthetic samples
SVM_acc_nosynth, SVM_auc_nosynth, SVM_f1_nosynth = acc_auc_roc_SVM(SVM_model, X_val_norm, y_val_norm)
rf_acc_nosynth, rf_auc_nosynth, rf_f1_nosynth = acc_auc_roc_general(rf_model, X_val_norm, y_val_norm)
xgb_acc_nosynth, xgb_auc_nosynth, xgb_f1_nosynth = acc_auc_roc_general(xgb_model, X_val_norm, y_val_norm)
knn_acc_nosynth, knn_auc_nosynth, knn_f1_nosynth = acc_auc_roc_general(knn_model, X_val_norm, y_val_norm)

# Save F1-score values for further use (load_and_plot.py)
with open("svm_f1.txt", "wb") as svm:
    pickle.dump(SVM_f1_nosynth, svm)
svm.close()
with open("rf_f1.txt", "wb") as rf:
    pickle.dump(rf_f1_nosynth, rf)
rf.close()
with open("xgb_f1.txt", "wb") as xgb:
    pickle.dump(xgb_f1_nosynth, xgb)
xgb.close()
with open("knn_f1.txt", "wb") as knn:
    pickle.dump(SVM_f1_nosynth, knn)
knn.close()

# Save "sizes_balance1" variables for further use (load_and_plot.py)
with open("sizes.txt", "wb") as siz:
    pickle.dump(sizes_balance1, siz)
siz.close()

# Generates all the previously indicated number of synthetic data samples 
for i in range(len(sizes_keys)):

    # Different iterations to evaluate variability of iterations 
    for j in range(aug_iterations):
          
        # Data augmentation WITHOUT splitting between controls and cases 
        for group in no_split : 

            # Synthetic data generation with CTGAN and Gaussian Copula 
            if group[0] == augmen2 : 
                mixed_data, synthetic_data = data_aug(gc, group, i, j)
            elif group[0] == augmen1 :
                mixed_data, synthetic_data = data_aug(ctgan, group, i, j)  
            
            # The case where only synthetic data is used to train the model
            if sizes_keys[i] == "only-synth":
                mixed_data = synthetic_data
            
            # Performs replacement and one-hot enconding in the training set without synthetic samples
            mixed_data = replacement(mixed_data)
            mixed_data = one_hot_enc(mixed_data)
            
            # Drop the same columns than before in the training set 
            mixed_data = mixed_data.drop([('Pulmonar Cancer',)],axis=1)
            mixed_data = mixed_data.drop([('Neuro Epilepsia',)],axis=1)
            mixed_data = mixed_data.drop([('Renal Otras',)],axis=1)

            # Saving the key combining balancing and augmentation technique strings
            method = group[1] + " + " + group[0] 

            # Compute metrics 
            PCD_val = PCD(mixed_data, train_data)
            KLD_val, _, _ = KLD(mixed_data.loc[:, mixed_data.columns != y_tag], train_PMFs, train_hist_bases)
            MMD_val = mmd_linear(X_train.to_numpy(), mixed_data.loc[:, mixed_data.columns != y_tag].to_numpy())
            ratio = (mixed_data[y_tag][mixed_data[y_tag] == 1].value_counts()[1])/(mixed_data[y_tag][mixed_data[y_tag] == 0].value_counts()[0])
            
            # Store metrics in dictionary 
            sdg_metrics[method][sizes_keys[i]]['PCD'][j] = PCD_val 
            sdg_metrics[method][sizes_keys[i]]['KLD'][j] = KLD_val 
            sdg_metrics[method][sizes_keys[i]]['MMD'][j] = MMD_val 
            sdg_metrics[method][sizes_keys[i]]['ratio'][j] = ratio 
            
            # Standardization
            X_norm, y_norm = standardization_cat(mixed_data.loc[:, mixed_data.columns != y_tag], mixed_data[y_tag], numerical_features)

            # ML models train 
            SVM_model, SVM_train_results, SVM_cv_results  = model_train(svm_model, svm_params, "SVM", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            rf_model, rf_train_results, rf_cv_results = model_train(rf_model, rf_params, "RF", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            xgb_model, xgb_train_results, xgb_cv_results = model_train(xgb_model, xgb_params, "XGB", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            knn_model, knn_train_results, knn_cv_results = model_train(knn_model, knn_params, "KNN", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            
            # Saving hyperparameters in the corresponding dictionary 
            hyperparameters['SVM'][method][sizes_keys[i]]['kernel'][j] = SVM_model.kernel
            hyperparameters['SVM'][method][sizes_keys[i]]['C'][j] = SVM_model.C
            hyperparameters['SVM'][method][sizes_keys[i]]['gamma'][j] = SVM_model.gamma

            hyperparameters['RF'][method][sizes_keys[i]]['estimators'][j] = rf_model.n_estimators
            hyperparameters['RF'][method][sizes_keys[i]]['max_feat'][j] = rf_model.max_features
            
            hyperparameters['XGB'][method][sizes_keys[i]]['estimators'][j] = xgb_model.n_estimators
            hyperparameters['XGB'][method][sizes_keys[i]]['lr'][j] = xgb_model.learning_rate

            hyperparameters['KNN'][method][sizes_keys[i]]['neigh'][j] = knn_model.n_neighbors
            hyperparameters['KNN'][method][sizes_keys[i]]['C'][j] = knn_model.weights

            # ML models' evaluation 
            SVM_acc, SVM_auc, SVM_f1 = acc_auc_roc_SVM(SVM_model, X_val_norm, y_val_norm)
            rf_acc, rf_auc, rf_f1 = acc_auc_roc_general(rf_model, X_val_norm, y_val_norm)
            xgb_acc, xgb_auc, xgb_f1 = acc_auc_roc_general(xgb_model, X_val_norm, y_val_norm)
            knn_acc, knn_auc, knn_f1 = acc_auc_roc_general(knn_model, X_val_norm, y_val_norm)
            
            # Store results in correspondant dictionary
            class_metrics['SVM'][method][sizes_keys[i]]['acc'][j] = SVM_acc
            class_metrics['SVM'][method][sizes_keys[i]]['auc'][j] = SVM_auc
            class_metrics['SVM'][method][sizes_keys[i]]['f1'][j] = SVM_f1

            class_metrics['RF'][method][sizes_keys[i]]['acc'][j]= rf_acc
            class_metrics['RF'][method][sizes_keys[i]]['auc'][j] = rf_auc
            class_metrics['RF'][method][sizes_keys[i]]['f1'][j] = rf_f1

            class_metrics['XGB'][method][sizes_keys[i]]['acc'][j] = xgb_acc
            class_metrics['XGB'][method][sizes_keys[i]]['auc'][j] = xgb_auc
            class_metrics['XGB'][method][sizes_keys[i]]['f1'] = xgb_f1

            class_metrics['KNN'][method][sizes_keys[i]]['acc'][j] = knn_acc
            class_metrics['KNN'][method][sizes_keys[i]]['auc'][j] = knn_auc
            class_metrics['KNN'][method][sizes_keys[i]]['f1'][j] = knn_f1
        
        # Data augmentation AFTER SPLITTING bewteen controls and cases  
        #### NOTE that CONDITIONS are only applied in CTGAN augmentation
        # due to its poor performance without fixing them, whereas Gaussian Copula
        # generates synthetic control or cases samples from the control and cases
        # datasets, respectively, without the need of fixing the output variable. 

        for group in split :
        
            # Synthetic data generation with CTGAN and Gaussian Copula 
            if group[0] == augmen2 : 
                mixed_data, synthetic_data = data_aug_after_split(gc, group, i, j)
            elif group[0] == augmen1 :
                mixed_data, synthetic_data = data_aug_cond_after_split(ctgan, group, [cond_negative, cond_positive], i, j)  
            
            # The case where only synthetic data is used to train the model
            if sizes_keys[i] == "only-synth":
                mixed_data = synthetic_data
            
            # Performs replacement and one-hot enconding in the training set without synthetic samples
            mixed_data = replacement(mixed_data)
            mixed_data = one_hot_enc(mixed_data)
            
            # Drop the same columns than before in the training set 
            mixed_data = mixed_data.drop([('Pulmonar Cancer',)],axis=1)
            mixed_data = mixed_data.drop([('Neuro Epilepsia',)],axis=1)
            mixed_data = mixed_data.drop([('Renal Otras',)],axis=1)

            # Saving the key combining balancing and augmentation technique strings
            method = group[1] + " + Sep. + " + group[0] 
            
            # Compute metrics 
            PCD_val = PCD(mixed_data, train_data)
            KLD_val, _, _ = KLD(mixed_data.loc[:, mixed_data.columns != y_tag], train_PMFs, train_hist_bases)
            MMD_val = mmd_linear(X_train.to_numpy(), mixed_data.loc[:, mixed_data.columns != y_tag].to_numpy())
            ratio = (mixed_data[y_tag][mixed_data[y_tag] == 1].value_counts()[1])/(mixed_data[y_tag][mixed_data[y_tag] == 0].value_counts()[0])
            
            # Store metrics in dictionary 
            sdg_metrics[method][sizes_keys[i]]['PCD'][j] = PCD_val 
            sdg_metrics[method][sizes_keys[i]]['KLD'][j] = KLD_val 
            sdg_metrics[method][sizes_keys[i]]['MMD'][j] = MMD_val 
            sdg_metrics[method][sizes_keys[i]]['ratio'][j] = ratio 
            
            # Standardization
            X_norm, y_norm = standardization(mixed_data.loc[:, mixed_data.columns != y_tag], mixed_data[y_tag])

            # ML models train 
            SVM_model, SVM_train_results, SVM_cv_results  = model_train(svm_model, svm_params, "SVM", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            rf_model, rf_train_results, rf_cv_results = model_train(rf_model, rf_params, "RF", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            xgb_model, xgb_train_results, xgb_cv_results = model_train(xgb_model, xgb_params, "XGB", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            knn_model, knn_train_results, knn_cv_results = model_train(knn_model, knn_params, "KNN", method, X_norm, y_norm, cv = 10, scoring = 'f1')
            
            # Saving hyperparameters in the corresponding dictionary 
            hyperparameters['SVM'][method][sizes_keys[i]]['kernel'][j] = SVM_model.kernel
            hyperparameters['SVM'][method][sizes_keys[i]]['C'][j] = SVM_model.C
            hyperparameters['SVM'][method][sizes_keys[i]]['gamma'][j] = SVM_model.gamma

            hyperparameters['RF'][method][sizes_keys[i]]['estimators'][j] = rf_model.n_estimators
            hyperparameters['RF'][method][sizes_keys[i]]['max_feat'][j] = rf_model.max_features
            
            hyperparameters['XGB'][method][sizes_keys[i]]['estimators'][j] = xgb_model.n_estimators
            hyperparameters['XGB'][method][sizes_keys[i]]['lr'][j] = xgb_model.learning_rate

            hyperparameters['KNN'][method][sizes_keys[i]]['neigh'][j] = knn_model.n_neighbors
            hyperparameters['KNN'][method][sizes_keys[i]]['C'][j] = knn_model.weights

            # ML models' evaluation 
            SVM_acc, SVM_auc, SVM_f1 = acc_auc_roc_SVM(SVM_model, X_val_norm, y_val_norm)
            rf_acc, rf_auc, rf_f1 = acc_auc_roc_general(rf_model, X_val_norm, y_val_norm)
            xgb_acc, xgb_auc, xgb_f1 = acc_auc_roc_general(xgb_model, X_val_norm, y_val_norm)
            knn_acc, knn_auc, knn_f1 = acc_auc_roc_general(knn_model, X_val_norm, y_val_norm)
            
            # Store results in correspondant dictionary
            class_metrics['SVM'][method][sizes_keys[i]]['acc'][j] = SVM_acc
            class_metrics['SVM'][method][sizes_keys[i]]['auc'][j] = SVM_auc
            class_metrics['SVM'][method][sizes_keys[i]]['f1'][j] = SVM_f1

            class_metrics['RF'][method][sizes_keys[i]]['acc'][j] = rf_acc
            class_metrics['RF'][method][sizes_keys[i]]['auc'][j] = rf_auc
            class_metrics['RF'][method][sizes_keys[i]]['f1'][j] = rf_f1

            class_metrics['XGB'][method][sizes_keys[i]]['acc'][j] = xgb_acc
            class_metrics['XGB'][method][sizes_keys[i]]['auc'][j] = xgb_auc
            class_metrics['XGB'][method][sizes_keys[i]]['f1'][j] = xgb_f1

            class_metrics['KNN'][method][sizes_keys[i]]['acc'][j] = knn_acc
            class_metrics['KNN'][method][sizes_keys[i]]['auc'][j] = knn_auc
            class_metrics['KNN'][method][sizes_keys[i]]['f1'][j] = knn_f1

# Calculate and store computation time of the framework
end = time.time()
total_time = end - start 

# Save results as dictionaries using pickle 
# Synthetic data metrics 
with open("sdg_metrics.pkl", "wb") as sdg_dict:
    pickle.dump(sdg_metrics, sdg_dict)
sdg_dict.close()

# Classification metrics 
with open("class_metrics.pkl", "wb") as class_metrics_dict:
    pickle.dump(class_metrics, class_metrics_dict)
class_metrics_dict.close()

# Hyperparameters 
with open("hyperparameters.pkl", "wb") as hp_dict:
    pickle.dump(hyperparameters, hp_dict)
hp_dict.close()

# FIGURE I - Scatter plots with trend line: Metrics vs. Data size
    
# Split CTGAN and Gaussian Copula methods to plot them separately
ctgan_combinations = [comb1, comb3, comb5, comb7]
gc_combinations = [comb2, comb4, comb6, comb8]
       
sizes = sizes_balance1
   
# Figure 
fig, axs = plt.subplots(3,2)
    
# Set IEEE style 
plt.style.use(['science','ieee'])

# CTGAN Plotting
for i in range(len(ctgan_combinations)):

   temp_pcd  = np.zeros(len(sizes_keys)) # variable to generate polyfit
   temp_mmd  = np.zeros(len(sizes_keys))
   temp_kld  = np.zeros(len(sizes_keys))

   for j in range(len(sizes_keys)):

      k = -1 # counter to -1 one to begin in 0

      for metric in mets :

         k = k + 1 # counter increments to draw the next cell

         scatter1 = axs[k,0].scatter(sizes[j], sdg_metrics[ctgan_combinations[i]][sizes_keys[j]][metric].mean(), color = ctgan_colors[i])
    
      temp_pcd[j] = sdg_metrics[ctgan_combinations[i]][sizes_keys[j]]['PCD'].mean()
      temp_mmd[j] = sdg_metrics[ctgan_combinations[i]][sizes_keys[j]]['MMD'].mean()
      temp_kld[j] = sdg_metrics[ctgan_combinations[i]][sizes_keys[j]]['KLD'].mean()
    
   # Calulate and draw the polynom
   z_pcd = np.polyfit(sizes, temp_pcd, 1)
   p_pcd = np.poly1d(z_pcd)

   z_mmd = np.polyfit(sizes, temp_mmd, 1)
   p_mmd = np.poly1d(z_mmd)

   z_kld = np.polyfit(sizes, temp_kld, 1)
   p_kld = np.poly1d(z_kld)

   # Line format must be specified different with orange colour
   line = ctgan_colors[i]+"--"
   axs[0,0].plot(sizes,p_pcd(sizes), line)
   axs[1,0].plot(sizes,p_mmd(sizes), line)
   axs[2,0].plot(sizes,p_kld(sizes), line)

# Gaussian Copula Plotting
for i in range(len(gc_combinations)):

   temp_pcd  = np.zeros(len(sizes_keys)) # variable to generate polyfit
   temp_mmd  = np.zeros(len(sizes_keys))
   temp_kld  = np.zeros(len(sizes_keys))

   for j in range(len(sizes_keys)):

      k = -1 # counter to -1 one to begin in 0

      for metric in mets :

         k = k + 1 # counter increments to draw the next cell

         scatter2 = axs[k,1].scatter(sizes[j], sdg_metrics[gc_combinations[i]][sizes_keys[j]][metric].mean(), color = gc_colors[i])
    
      temp_pcd[j] = sdg_metrics[gc_combinations[i]][sizes_keys[j]]['PCD'].mean()
      temp_mmd[j] = sdg_metrics[gc_combinations[i]][sizes_keys[j]]['MMD'].mean()
      temp_kld[j] = sdg_metrics[gc_combinations[i]][sizes_keys[j]]['KLD'].mean()
    
   # Calulate and draw the polynom
   z_pcd = np.polyfit(sizes, temp_pcd, 1)
   p_pcd = np.poly1d(z_pcd)

   z_mmd = np.polyfit(sizes, temp_mmd, 1)
   p_mmd = np.poly1d(z_mmd)

   z_kld = np.polyfit(sizes, temp_kld, 1)
   p_kld = np.poly1d(z_kld)

   # Line format must be specified different with orange colour
   axs[0,1].plot(sizes,p_pcd(sizes), c = gc_colors[i], ls = "--")
   axs[1,1].plot(sizes,p_mmd(sizes), c = gc_colors[i], ls = "--")
   axs[2,1].plot(sizes,p_kld(sizes), c = gc_colors[i], ls = "--")

# Remove x-labels
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])

# Set figure text
fig.text(0.5, 0.04, 'Nº of samples', ha='center')
fig.text(0.02, 0.75, 'PCD', va='center', rotation='vertical')
fig.text(0.02, 0.5, 'MMD', va='center', rotation='vertical')
fig.text(0.02, 0.25, 'KLD', va='center', rotation='vertical')

# Set legend
axs[0,0].legend(ctgan_combinations, bbox_to_anchor=(-0.25,1.02,1,0.2), loc="lower left",
                mode="None", borderaxespad=0, ncol=2, prop={'size': 4})
axs[0,1].legend(gc_combinations, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="None", borderaxespad=0, ncol=2, prop={'size': 4})

name = dataset_name + "_metrics_vs_synthetic_data_samples"
plt.savefig(name , dpi=600)

# FIGURE II - F1-Score versus data samples (Best abd worst cases) 

best_worst = ['Borderline + Sep. + CTGAN', 'NC + CTGAN']  

fig, ax = plt.subplots(2)

# Set IEEE style 
plt.style.use(['science','ieee'])

# Iterating the dictionary to plot the correspondant contents   
for m in range(len(best_worst)) :  
    
    for i in range(len(models)):
        
        x_vector = np.zeros(len(sizes_keys)) # Vector to fill before plotting the errorbar
        y_vector = np.zeros(len(sizes_keys))
        err_vector = np.zeros(len(sizes_keys))
        
        for method in best_worst:
            
            for j in range(len(sizes_keys)):

                x_vector[j] = sizes[j]
                y_vector[j] = class_metrics[models[i]][best_worst[m]][sizes_keys[j]]['f1'].mean()
                err_vector[j] = class_metrics[models[i]][best_worst[m]][sizes_keys[j]]['f1'].std()

        ax[m].errorbar(x_vector, y_vector, err_vector, capsize = 4.0, linestyle=':', marker='o', color=model_colors[i])

# Set figure text 
fig.text(0.5, 0.04, 'Nº of samples', ha='center')
fig.text(0.01, 0.5, 'F1-score', va='center', rotation='vertical')

# Write the name of the chosen methods
fig.text(0.20, 0.15, best_worst[0])
fig.text(0.20, 0.55, best_worst[1])

# Remove x-labels
ax[0].set_xticklabels([])

# Set legend 
ax[0].legend(models, bbox_to_anchor=(0.07,1.02,1,0.2), loc="lower left",
                mode="None", borderaxespad=0, ncol=4, prop={'size': 6})

# Plot the reference lines (Validation results without synthetic data)
ax[0].axhline(y=SVM_f1_nosynth, color='b', linestyle='--')  
ax[0].axhline(y=rf_f1_nosynth, color='r', linestyle='--') 
ax[0].axhline(y=xgb_f1_nosynth, color='k', linestyle='--') 
ax[0].axhline(y=knn_f1_nosynth, color='g', linestyle='--')  
ax[1].axhline(y=SVM_f1_nosynth, color='b', linestyle='--')  
ax[1].axhline(y=rf_f1_nosynth, color='r', linestyle='--') 
ax[1].axhline(y=xgb_f1_nosynth, color='k', linestyle='--') 
ax[1].axhline(y=knn_f1_nosynth, color='g', linestyle='--')              

name = dataset_name + "_f1_vs_data_samples"
plt.savefig(name, dpi = 600)

# FIGURE III: Metrics vs. F1-Score

# Best combination: ADASYN + GC
best_method = "Borderline + Sep. + CTGAN"

fig, ax = plt.subplots(3)

plt.style.use(['science','ieee'])

for i in range(len(models)): 

    temp_f1  = np.zeros(len(sizes_keys)) # variable to generate polyfit
    temp_pcd  = np.zeros(len(sizes_keys)) 
    temp_mmd  = np.zeros(len(sizes_keys)) 
    temp_kld  = np.zeros(len(sizes_keys)) 

    for j in range(len(sizes_keys)): 
        
        for k in range(len(mets)):
            
            scatter1 = ax[k].scatter(class_metrics[models[i]][best_method][sizes_keys[j]]['f1'].mean(), 
                        sdg_metrics[best_method][sizes_keys[j]][mets[k]].mean(),
                        color = model_colors[i])            
    
        temp_f1[j] = class_metrics[models[i]][best_method][sizes_keys[j]]['f1'].mean()
        temp_pcd[j] = sdg_metrics[best_method][sizes_keys[j]]['PCD'].mean()
        temp_mmd[j] = sdg_metrics[best_method][sizes_keys[j]]['MMD'].mean()
        temp_kld[j] = sdg_metrics[best_method][sizes_keys[j]]['KLD'].mean()
    
    line = model_colors[i]+"--"

    z_pcd = np.polyfit(temp_f1, temp_pcd, 1)
    p_pcd = np.poly1d(z_pcd)
    ax[0].plot(temp_f1,p_pcd(temp_f1), line)

    z_mmd = np.polyfit(temp_f1, temp_mmd, 1)
    p_mmd = np.poly1d(z_mmd)
    ax[1].plot(temp_f1,p_mmd(temp_f1), line)

    z_kld = np.polyfit(temp_f1, temp_kld, 1)
    p_kld = np.poly1d(z_kld)
    ax[2].plot(temp_f1,p_kld(temp_f1), line)

# Set figure text 
fig.text(0.5, 0.04, 'F1-Score', ha='center')
fig.text(0.01, 0.75, 'PCD', va='center', rotation='vertical')
fig.text(0.01, 0.5, 'MMD', va='center', rotation='vertical')
fig.text(0.01, 0.25, 'KLD', va='center', rotation='vertical')

# Remove x-labels
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])

# Set legend 
ax[0].legend(models, bbox_to_anchor=(0.07,1.02,1,0.2), loc="lower left",
                mode="None", borderaxespad=0, ncol=4, prop={'size': 6})

name = dataset_name + "_sdg_metrics_vs_f1_score"
plt.savefig(name, dpi=600)