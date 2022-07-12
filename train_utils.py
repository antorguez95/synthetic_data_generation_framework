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
from sklearn import base
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing

import pandas as pd

from typing import Dict, List, Tuple

def model_train(model : base.BaseEstimator, params : Dict, model_name: str, 
              sdg_technique : str, X : pd.DataFrame , Y : pd.DataFrame,  cv : int = 10,
              scoring : str = 'f1') -> Tuple[base.BaseEstimator, pd.DataFrame, pd.DataFrame] :  
       """Trains a model using Grid Search hyperparameters optimization strategy.
    
       Args:
       -----
              model: sklearn estimator instance.
              params: Hyperparameters to be tuned (model dependent).
              model_name: ML model name. 
              sdg_technique : Synthetic Data Generation employed to generate the synthetic 
              dataset used to train the ML model
              X: The features/independent variables of a given dataset.
              Y: The target/independent variable of a given dataset
              cv: cross-validation splitting strategy. Defaults to 10, 
              (more in https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
              scoring : target metric to evaluate the performance of croass-validated model. Defaults to 'f1', 
              since this framework is thought to work
              with imbalanced datasets. 
              (more in https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

       Returns:
       --------
              best_model: estimator that shows best cross-validation performance.
              results: summary of best estimator results. 
              overall_results: summary of overall results.
       """
       
       # Grid Search hyperparameters optimization 
       grid = GridSearchCV(model,
                        params,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = -1,
                        return_train_score = True,
                        verbose = 0).fit(X,Y)

       a = grid.cv_results_
       b = pd.DataFrame.from_dict(a, orient='columns')
       
       # Dropping worthless columns 
       c = b.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 
                   'std_score_time','split0_test_score',
                   'split1_test_score', 'split2_test_score', 'split3_test_score',
                   'split4_test_score', 'split5_test_score', 'split6_test_score',
                   'split7_test_score', 'split8_test_score', 'split9_test_score',
                   'split0_train_score', 'split1_train_score', 'split2_train_score',
                   'split3_train_score', 'split4_train_score', 'split5_train_score',
                   'split6_train_score', 'split7_train_score', 'split8_train_score',
                   'split9_train_score', 'mean_train_score', 'std_train_score'], axis = 1)
       
       # Save results from best test score to worst
       overall_results = c.sort_values('rank_test_score')

       # Save best model 
       best_model = grid.best_estimator_
       
       # Cross validation score of the best model 
       results = [cross_val_score(best_model, X, Y, cv = cv, scoring=scoring).mean(), cross_val_score(model, X, Y, cv = cv, scoring=scoring).std()] 
       
       # Printing training sets results 
       print(model_name ,": ", sdg_technique," -> ", scoring, " on training set is", results[0],"(",results[1],")")
       
       return best_model, results, overall_results

def standardization(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame] :

       """Performs standardization by removing the mean and scling to unit variance
       in the numerical variables that belongs to a dataset that also has categorical 
       and/or binary variables. See 
       https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
       for more information. 

       Args:
       -----
       X: features of the dataset.
       y: target variable. 
       numerical_variables : list containing the numerical features of the dataset to be standardized

       Returns:
       --------
       X: standardized features .
       y: target variable. 
       """  

       # From DataFrame to numpy array
       X = X.to_numpy()
       y = y.to_numpy()
       y = y.ravel() # To avoid warning. We go from a column vector to a 1D-array

       # Standardization
       X_norm = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

       return X_norm, y

def standardization_cat(X : pd.DataFrame, y : pd.DataFrame, 
                     numerical_variables : List) -> Tuple[pd.DataFrame, pd.DataFrame]  :

       """Performs standardization by removing the mean and scling to unit variance
       in the numerical variables that belongs to a dataset that also has categorical 
       and/or binary variables. See 
       https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
       for more information. 
    
       Args:
       -----
              X: features of the dataset.
              y: target variable. 
              numerical_variables : list containing the numerical features of the dataset to be standardized

       Returns:
       --------
              X: standardized features .
              y: target variable. 
       """
       
       # From DataFrame to numpy array
       y = y.to_numpy()
       y = y.ravel() # To avoid warning. We go from a column vector to a 1D-array

       # Standardization only on numerical variables
       X[numerical_variables] = preprocessing.StandardScaler().fit(X[numerical_variables]).transform(X[numerical_variables].astype(float))
       X_norm = X
       
       return X_norm,y
