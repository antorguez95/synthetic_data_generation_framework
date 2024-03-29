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

import numpy as np

import pandas as pd

from sklearn import metrics
from sklearn.metrics import plot_roc_curve, f1_score
from sklearn import base

import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

def acc_auc_roc_SVM(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):
      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a given SVM Classifier and a given 
      dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
            X : features of the dataset
            y : target variable of the dataset.

      Returns:
      --------
            acc: accuracy on the validation set.
            roc_auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 
      
      # Accuracy 
      acc = metrics.accuracy_score(y, y_pred)
 
      # AUC
      y_score = model.decision_function(X)
      fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
      roc_auc = metrics.auc(fpr, tpr)
         
      # F1 Score
      f1_sc = f1_score(y, y_pred)
        
      return acc, roc_auc, f1_sc
		
def acc_auc_roc_general(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):
      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a sklearn Classifier and a given 
      dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn classifier instance.
            X : features of the dataset
            y : target variable of the dataset. 
      Returns:
      --------
            acc: accuracy on the validation set.
            auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 
      
      # Accuracy
      acc = metrics.accuracy_score(y, y_pred)
                       
      # ROC
      disp = plot_roc_curve(model, X, y)
      
      # AUC
      auc = disp.roc_auc

      # F1 Score
      f1_sc = f1_score(y, y_pred)
              
      return acc, auc, f1_sc 
    
def get_eval_dictionaries(sdg_combinations: List, sizes_keys: List, class_metrics : List, iterations : int) -> Tuple [Dict, Dict, Dict]:
      """This function generates and returns three different dictionaries to evaluate
       the Synthetic Data Generation (SDG) in three ways:
            1) Statistical metrics of SDG: Pairwise Correlation Difference (PCD), 
            Kullback-Leibler Divergence (KLD), Maximun Mean Discrepancy (MMD) and
            Case/Control ratio. 
            2) Classification Performance employing four different Machine Learning
            (ML) models: SVM, RF, XGB and KNN. Notice that any changes on the 
            framework must be changed in this code for a proper work of it. 
            3) Hyperparameters optimization. These are ML models-dependent. As 
            abovementioned, changes on tuned parameters or in the models
            themselves, must be translated into changes in this code. 

      Args:
      -----
            sdg_combinations: list containing the employed SDG combinations.
            sizes_keys: identificators of the different amount of synthetic samples generated. 
            class_metrics: list containing the metrics used to evaluated the ML models 
            iterations: number of synthetic data generations iterations to be evaluated 

      Returns:
      --------
            sdg_metrics: dictionary of structure {SDG combinations : 
                                                      {different generated sizes:  
                                                            {statistical metrics : np.arrray(iterations)}}} 
            class_metrics: dictionary of structure {ML model: 
                                                      {SDG combinations : 
                                                            {different generated sizes:  
                                                                  {classification metrics : np.array(iterations)}}}
            hyperparameters: dictionary of structure {ML model: 
                                                      {SDG combinations : 
                                                            {different generated sizes:  
                                                                  {best model hyperparameters : list(iterations)}}}.
      """           
                  
                  
      # Dictionary to store metrics for all synthetic data generation combinations 
      sdg_metrics = {}

      # Dictionary structure: {SDG combinations : {different generated sizes:  {statistical metrics : np.arrray(iterations)}}}   
      for comb in sdg_combinations : 
            sdg_metrics[comb] = {'quarter': {}, 'half': {}, 
                              'unit':{}, 'double': {}, 'quadruple' : {}}
            for sizes in sizes_keys : 
                  sdg_metrics[comb][sizes] = {'PCD': np.zeros(iterations), 'MMD': np.zeros(iterations), 
                        'KLD':np.zeros(iterations), 'ratio': np.zeros(iterations)}


      # In order to compare results obtained with SDG to the reference, train is added as a key in the dictionary
      train = "No synthetic"
      sdg_combinations.insert(0, train)

      # Dictionary II to store classification metrics for all synthetic data generation combinations and all ML classifiers
      class_metrics = {'SVM' : {}, 'RF' : {}, 'XGB' : {}, 'KNN': {}}

      # Dictionary structure: {ML model: {SDG combinations : {different generated sizes:  {classification metrics : np.array(iterations)}}}
      for key in class_metrics :    
                  for comb in sdg_combinations : 
                        class_metrics[key][comb] = {'quarter': {}, 'half': {}, 
                                          'unit':{}, 'double': {}, 'quadruple' : {}}
                        for sizes in sizes_keys : 
                              class_metrics[key][comb][sizes] = {'acc': np.zeros(iterations), 'auc': np.zeros(iterations), 
                                                            'f1':np.zeros(iterations)}

      # Dictionary III to store optimized models after GridSearch Hiperparametrization
      hyperparameters = {'SVM' : {}, 'RF' : {}, 'XGB' : {}, 'KNN': {}}

      # Dictionary structure: {ML model: {SDG combinations : {different generated sizes:  {best model hyperparameters : list(iterations)}}}
      # Notice that hyperparameters are ML model-specific 
      for key in hyperparameters :    
                  for comb in sdg_combinations : 
                        hyperparameters[key][comb] = {'quarter': {}, 'half': {}, 
                                          'unit':{}, 'double': {}, 'quadruple' : {}}
      for comb in sdg_combinations : 
                  for sizes in sizes_keys : 
                        hyperparameters['SVM'][comb][sizes] = {'kernel': [None]*iterations, 'C': [None]*iterations, 'gamma' : [None]*iterations}
                        hyperparameters['RF'][comb][sizes] = {'estimators': [None]*iterations, 'max_feat': [None]*iterations}
                        hyperparameters['XGB'][comb][sizes] = {'estimators': [None]*iterations, 'lr': [None]*iterations}
                        hyperparameters['KNN'][comb][sizes] = {'neigh': [None]*iterations, 'C': [None]*iterations, 'weights' : [None]*iterations}

      return sdg_metrics, class_metrics, hyperparameters