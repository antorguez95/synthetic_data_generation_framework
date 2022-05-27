
# Dependencies 
import pandas as pd  
import numpy as np  

from typing import Callable, Tuple, Dict, List 

from imblearn.over_sampling  import ADASYN, SMOTE, SMOTENC, KMeansSMOTE, SVMSMOTE, BorderlineSMOTE

import matplotlib.pyplot as plt

from sdv.constraints.base import Constraint
from sdv.sampling import Condition 

import scipy.stats
import numpy as np

from sklearn import base

import pickle 

import os

def PCD(X, X_ref):
      dif_corr = X.corr() - X_ref.corr()
      from numpy import linalg
      value = linalg.norm(dif_corr, ord = 'fro')
      return value 

def PMF(X):  
  PMFs = {}
  bases = {}
  i = 0
  for feature in X :
    # calculate histograms
    bins = np.arange(np.floor(X[feature].min()),np.ceil(X[feature].max()+0.01),step=0.1) 
    value, base = np.histogram(X[feature], bins = bins, density = 1) 
    bases[feature] = base
    hist = scipy.stats.rv_histogram([value, base]) 
    # Calculate PMF
    PMFs[feature]= hist.pdf(base)
    i = i+1

  return PMFs, bases
 	  
def kl_divergence(p, q):
    p[p == 0] = 0.0000001
    q[q == 0] = 0.0000001
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def KLD(X, ref_PMF, ref_hist_base): 
  PMFs = {}
  KLDs = {}
  i = 0
  for feature in X :
    # calculate histograms
    bins = np.arange(np.floor(X[feature].min()),np.ceil(X[feature].max()+0.01),step=0.1) 
    value, base = np.histogram(X[feature], bins = bins, density = 1)  
    hist = scipy.stats.rv_histogram([value, base]) 
    # Calculate PMF
    PMF = hist.pdf(ref_hist_base[feature])
    PMFs[feature] = PMF
    # Calculate KLD
    KLDs[feature] = kl_divergence(ref_PMF[feature], PMF)
    i = i+1 

    KLD_total = sum(KLDs.values())
    
  return KLD_total, KLDs, PMFs
	
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def balancing_eval(dataset : str, X: pd.DataFrame, y: pd.DataFrame, ref_data : pd.DataFrame,
                 columns: List, y_tag : str, algorithms: Tuple [str, str, str, str, str, str], 
                 cat_features_index : List, filename : str = "balancing_metrics.csv" , iterations : int = 100,
                 store_path : str = r"./results") -> None:    
    
    """Performs an evaluation of different balancing algorihtms with a given datasets.
    This evaluation iterates an specific number of times to assess the variability of 
    these algorithms (defaults to 100). Pairwise Correlation Difference (PCD) evaluates
    the replication of linear correlations. Kullback-Leibler Divergence (KLD) and
    Maximum Mean Discrepancy (MMD) evaluates features distribution similarity at 
    feature-level and dataset-level, respectively. Results are plotted using boxplots
    and stored in a .csv. 

    Notice that for fairly balanced datasets, some algorithms may faile or not converge. When this happens,
    set one of the strings in the algorithm Tuple to None, so this algorithm is not analysed. 

    Args:
    -----
        dataset: name of the dataset to store the figures 
        X: features of the dataset to be balance
        y: target variable taken as a reference to balance the dataset
        ref_data: dataset before balancing step to be compared with and compute metrics 
        columns: list with a name of the features of the dataset
        y_tag: name of the target variable 
        algorithms: balancing algorithms to be evaluated. Algorithms better be placed in this order as a tuple:
        ["ADASYN", "SMOTE", "SMOTENC", "KMeansSMOTE", "SVMSMOTE", "BorderlineSMOTE"]. To ommit an algorithm evaluation,
        substitute the given algorithm by "None". To further configuration of these algorithms, manage them into their
        correspondant function. 
        filename: name of the .csv file to store the results 
        iterations: number of iterations in the algorithm evaluation 
        store_path: path to store obtained results 

    Returns:
    --------
        None
    """

    # Colors used in the boxplots
    colors = ['red', 'green', 'grey', 'blue','black', 'brown']

    # List to be filled with algorithms indexes equal to None
    none_idxs = []

    # Generator to acquire indexes where algorithms is None
    none_idx_gen = (i for i,value in enumerate(algorithms) if value == None)

    # Save indices where algorithms is None
    for i in none_idx_gen:
        none_idxs.append(i)

    # Delete colors that corresponds to the algorihtms equal to None 
    for ele in sorted(none_idxs, reverse = True):
        del colors[ele]
    
    # Remove None elements from list 
    for element in algorithms: 
        if element == None:
            algorithms.remove(None)
    
    # Empty dictionary to store metrics 
    metrics = {}
    
    # Declaration of empty numpy arrays to store metrics
    for algorithm in algorithms : 
            metrics[algorithm] = {'PCD': np.zeros(iterations), 'MMD': np.zeros(iterations), 
                                'KLD':np.zeros(iterations), 'ratio': np.zeros(iterations)}
     
    # Perform analysis on given iterations 
    for i in range(iterations):  
        
        # Iterate over selected algorithms
        for algorithm in algorithms :        
                
                # Generating synthetic balancing for each algorithm 
                if algorithm == 'ADASYN': 
                    X_res, y_res = ADASYN(sampling_strategy = 'minority',
                                                                    random_state = None,
                                                                    n_neighbors = 5,
                                                                    n_jobs = None).fit_resample(X, y)                                                    
                elif algorithm == 'SMOTE':     
                    X_res, y_res = SMOTE(sampling_strategy = 'minority', 
                                                                random_state = None,
                                                                k_neighbors = 5,
                                                                n_jobs = None).fit_resample(X, y)   
                elif algorithm == 'SMOTENC':     
                    X_res, y_res = SMOTENC(categorical_features = cat_features_index,
                                                                  sampling_strategy = 'minority', 
                                                                  random_state = None,
                                                                  k_neighbors = 5,
                                                                  n_jobs = None).fit_resample(X, y)
                elif algorithm == 'KMeansSMOTE':
                    X_res, y_res = KMeansSMOTE(sampling_strategy = 'minority',
                                                                    random_state = None,
                                                                    k_neighbors = 5,
                                                                    n_jobs = None,
                                                                    kmeans_estimator = None,
                                                                    cluster_balance_threshold = "auto",
                                                                    density_exponent = "auto").fit_resample(X, y)
                elif algorithm == 'SVMSMOTE':
                    X_res, y_res = SVMSMOTE(sampling_strategy = 'minority',
                                                                    random_state = None,
                                                                    k_neighbors = 5,
                                                                    n_jobs = None,
                                                                    m_neighbors = 10,
                                                                    #svm_estimator = SVC(),
                                                                    out_step = 0.5).fit_resample(X, y)
                elif algorithm == 'BorderlineSMOTE':
                    X_res, y_res = BorderlineSMOTE(sampling_strategy = 'minority',
                                                                    random_state = None,
                                                                    k_neighbors = 5,
                                                                    n_jobs = None,
                                                                    m_neighbors = 10,
                                                                    kind = 'borderline-1').fit_resample(X, y)
                # X and Y in the same dataframe 
                X_res.reset_index(drop=True, inplace=True)
                y_res.reset_index(drop=True, inplace=True)
                data = pd.concat([X_res, y_res], axis = 1)

                # Compute and store metrics in dictionary 
                metrics[algorithm]['PCD'][i] = PCD(data, ref_data)
                metrics[algorithm]['MMD'][i] = mmd_linear(ref_data[columns].to_numpy(), data[columns].to_numpy())
                train_PMFs, train_hist_bases = PMF(ref_data) # Computes distribution to calculate KLD
                metrics[algorithm]['KLD'][i], _, _ = KLD(data[columns], train_PMFs, train_hist_bases) 
                metrics[algorithm]['ratio'][i] = (data[y_tag][data[y_tag] == 1].value_counts()[1])/(data[y_tag][data[y_tag] == 0].value_counts()[0])
                 
    
    # Generates and save boxplots of 4 metrics 
    # Store files in directory. If it does not exist, create it 
    if not os.path.exists(store_path):
            os.mkdir(store_path)

    # Go to given path 
    os.chdir(store_path)

    # From numpy to DataFrame to suit the calculation of boxplots
    # Remove "None" element from algorithms list 
    #algorithms.remove(None)

    # Empty list declaration 
    pcd = list()
    kld = list()
    mmd = list()
    ratio = list()

    # Iterate over the selected algorithms
    for algorithm in algorithms: 
    
        # Create list of metrics to convert it to dataFrame 
        pcd.append(metrics[algorithm]['PCD'])
        kld.append(metrics[algorithm]['KLD'])
        mmd.append(metrics[algorithm]['MMD'])
        ratio.append(metrics[algorithm]['ratio'])
        
    PCD_df = pd.DataFrame(np.array(pcd).T,
                            columns = [algorithms])
    KLD_df = pd.DataFrame(np.array(kld).T,
                            columns = [algorithms])
    MMD_df = pd.DataFrame(np.array(mmd).T,
                            columns = [algorithms])
    ratio_df = pd.DataFrame(np.array(ratio).T,
                            columns = [algorithms])

    # Generate and save boxplots 
    fig = plt.figure(figsize=(2.5,2.5), dpi=300, linewidth=5, tight_layout=True)
    plt.style.use(['science','ieee'])
    fig.text(0.0, 0.5, 'PCD', va='center', rotation='vertical')
    bplot1 = plt.boxplot(PCD_df,
                        vert=True,  
                        patch_artist=True)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for i in range(0,len(algorithms)):
                plt.setp(bplot1["boxes"][i], facecolor=colors[i])
                plt.setp(bplot1["fliers"][i], markeredgecolor=colors[i])
                plt.setp(bplot1["caps"][i], markeredgecolor=colors[i])
    plt.tick_params(labelbottom = False, bottom = False)
    name = dataset + '_PCD_boxplot' 
    plt.savefig(name, dpi=600)
    
    fig = plt.figure(figsize=(2.5,2.5), dpi=300, linewidth=1,tight_layout=True)
    plt.style.use(['science','ieee'])
    fig.text(0.0, 0.5, 'KLD', va='center', rotation='vertical')
    bplot2 = plt.boxplot(KLD_df,
                        vert=True,  # vertical box alignment
                        patch_artist=True)  # fill with color
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for i in range(0,len(algorithms)):
                plt.setp(bplot2["boxes"][i], facecolor=colors[i])
                plt.setp(bplot2["fliers"][i], markeredgecolor=colors[i])
                plt.setp(bplot2["caps"][i], markeredgecolor=colors[i])
    plt.tick_params(labelbottom = False, bottom = False)
    name = dataset + '_KLD_boxplot' 
    plt.savefig(name, dpi=600)
    
    fig = plt.figure(figsize=(2.5,2.5), dpi=300, linewidth=1, tight_layout=True)
    plt.style.use(['science','ieee'])
    fig.text(0.0, 0.5, 'MMD', va='center', rotation='vertical')
    bplot3 = plt.boxplot(MMD_df,
                        vert=True,  # vertical box alignment
                        patch_artist=True)  # fill with color
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for i in range(0,len(algorithms)):
                plt.setp(bplot3["boxes"][i], facecolor=colors[i])
                plt.setp(bplot3["fliers"][i], markeredgecolor=colors[i])
                plt.setp(bplot3["caps"][i], markeredgecolor=colors[i])
    plt.tick_params(labelbottom = False, bottom = False)
    name = dataset + '_MMD_boxplot' 
    plt.savefig(name, dpi=600)
    
    # Saving results on .csv
    f = open(filename,"w+") 
    f.write("Method:,PCD,MMD,KLD,ratio\n")
            
    # Loop to save results from selected algorithms
    for algorithm in algorithms : 
        f.write(algorithm +",%f(%f),%f(%f),%f(%f),%f(%f)\n" % (metrics[algorithm]['PCD'].mean(), metrics[algorithm]['PCD'].std(), metrics[algorithm]['KLD'].mean(), metrics[algorithm]['KLD'].std(), metrics[algorithm]['MMD'].mean(), metrics[algorithm]['MMD'].std(), metrics[algorithm]['ratio'].mean(), metrics[algorithm]['ratio'].std()))
    f.close()

def data_aug(model, augmenting_params : Tuple [str, str, pd.DataFrame, List], 
            size_index : int, iter : int = 0, SAVE_DATASETS : bool = False) -> Tuple [pd.DataFrame, pd.DataFrame]:
    
    """Generates synthetic samples given a Synthetic Generator model. The number of samples
    is given by the "num" variable. This function has only been used with Gaussian Copula and 
    CTGAN from SDV libraries. It shall work fine with other generators or it shall not.
    For more information, please refer to https://sdv.dev/SDV/.

    Args:
    -----
        model: synthetic data generator model (so far only CTGAN or Gaussian Copula from https://sdv.dev/SDV/
        have been tested).
        augmenting_params: Tuple [augmentation_technique, balancing_tecnique, data, num_samples]. First two elements
        are strings to properly store the dataset. Last elements are the dataframe containing the dataset
        itself and the list with the diffrent amount of synthetic samples to be generated. 
        Order must be as described to ensure proper functioning of this code. 
        augmentation_technique : name of the used data augmentation technique 
        balancing_technique : name of the used data balancing technique 
        data: reference dataset to fit the synthetic generator 
        conds: conditions to generate the dataset (e.g., set Glucose value over 100).
        size_index: when more than one size is tested, indicates the index of it.
        iter: synthetic data generation iteration index 
        SAVE_DATASETS: when set to True, all generated datasets are stored using pickle in ./synthetic_datasets. When False, 
        dataset is generated without being stored. 

    Returns:
    --------
        aug_data: dataframe containing synthetic generated samples together with real samples
        after shuffling. 
        samples: dataframe containing ONLY synthetic generated samples. 
    """
    
    # Fit Synthetic Data Generator to the original data 
    model.fit(augmenting_params[2])

    # Generates "num" synthetic samples 
    samples = model.sample(augmenting_params[3][size_index])
    
    # Join all the data in one dataframe
    augmenting_params[2].reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    aug_data = pd.concat([augmenting_params[2], samples])

    # Re-order instances (shuffle)
    aug_data = aug_data.sample(frac=1)

    # Store the dataset  
    if SAVE_DATASETS == True: 
        if not os.path.exists(r"./synthetic_datasets"):
            os.mkdir(r"./synthetic_datasets")
        os.chdir(r"./synthetic_datasets")
        filename = ("data_"+ augmenting_params[1] + "_" + augmenting_params[0] + "_size"+"%d"+"_iter_"+"%d"+".sav") % (size_index, iter)
        pickle.dump(aug_data, open(filename, 'wb')) 
        
        # Return to previous directory 
        os.chdir("..")  

    return aug_data, samples

def data_aug_cond(model, augmenting_params : Tuple [str, str, pd.DataFrame, List], 
                conds : Dict, size_index : int, iter : int = 0, SAVE_DATASETS : bool = False) -> Tuple [pd.DataFrame, pd.DataFrame]:

    """Generates synthetic samples given a Synthetic Generator model and certain conditions previously 
    defined through the use of a dictionarty. The number of samples is given by the "num" variable. 
    This function has only been used with Gaussian Copula and CTGAN from SDV libraries. 
    It shall work fine with other generators or it shall not.For more information, 
    please refer to https://sdv.dev/SDV/. It is esentially the same of "data_aug" function but adding
    the conditional generation. 

    Args:
    -----
        model: synthetic data generator model (so far only CTGAN or Gaussian Copula from https://sdv.dev/SDV/
        have been tested)-
        augmenting_params: Tuple [augmentation_technique, balancing_tecnique, data, num_samples]. First two elements
        are strings to properly store the dataset. Last elements are the dataframe containing the dataset
        itself and the list with the diffrent amount of synthetic samples to be generated. 
        Order must be as described to ensure proper functioning of this code.
        conds: conditions to generate the dataset (e.g., set Glucose value over 100).
        size_index: when more than one size is tested, indicates the index of it.
        iter: synthetic data generation iteration index  
        SAVE_DATASETS: when set to True, all generated datasets are stored using pickle in ./synthetic_datasets. When False, 
        dataset is generated without being stored. 
        

        
    Returns:
    --------
        aug_data: dataframe containing synthetic generated samples together with real samples
        after shuffling. 
        samples: dataframe containing ONLY synthetic generated samples. 
    """
    
    # Fit Synthetic Data Generator to the original data 
    model.fit(augmenting_params[2])

    # Creates condition for all generated rows
    condition = Condition(conds, num_rows=augmenting_params[3][size_index])
    
    # Generates "num" synthetic samples with certain conditions 
    samples = model.sample_conditions([condition], 100, randomize_samples = False)
    
    # Join all the data in one dataframe
    augmenting_params[2].reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    aug_data = pd.concat([augmenting_params[2], samples])

    # Re-order instances (shuffle)
    aug_data = aug_data.sample(frac=1)

    # Store the dataset  
    if SAVE_DATASETS == True: 
        if not os.path.exists(r"./synthetic_datasets"):
            os.mkdir(r"./synthetic_datasets")
        os.chdir(r"./synthetic_datasets")
        filename = ("data_"+ augmenting_params[1] + "_" + augmenting_params[0] + "_size"+"%d"+"_iter_"+"%d"+".sav") % (size_index, iter)
        pickle.dump(aug_data, open(filename, 'wb')) 
    
        # Return to previous directory 
        os.chdir("..") 
    
    return aug_data, samples

def data_aug_after_split(model, augmenting_params : Tuple [str, str, pd.DataFrame, pd.DataFrame, List], 
        size_index : int = 0, iter : int = 0, SAVE_DATASETS : bool = False) -> Tuple [pd.DataFrame, pd.DataFrame]:
    
    """Generates synthetic samples given a Synthetic Generator model after splitting the dataset
    into two sub-datasets containing only controls (e.g., Non-Diabetic patient) and cases 
    (e.g., Diabetic patients). Data generation is performed separately and the the two subdatasets
    are then joint. The number of generated samples is given by the "num" variable. This function has
    only been used with Gaussian Copula and CTGAN from SDV libraries. It shall work fine with other 
    generators or it shall not. For more information, please refer to https://sdv.dev/SDV/.

    Args:
    -----
        model: synthetic data generator model (so far only CTGAN or Gaussian Copula from https://sdv.dev/SDV/
        have been tested)
        augmenting_params: Tuple [augmentation_technique, balancing_tecnique, controls, cases, num_samples]. 
        First two elements are strings to properly handle and store the dataset. Next elements are the dataframes 
        containing the controls and cases datasets, respectively. The last element is a list with the diffrent 
        amounts of synthetic samples to be generated. Order must be as described to ensure proper functioning of this code. 
        size_index: when more than one size is tested, indicates the index of it.
        iter: synthetic data generation iteration index  
        SAVE_DATASETS: when set to True, all generated datasets are stored using pickle in ./synthetic_datasets. When False, 
        dataset is generated without being stored. 

    Returns:
    --------
        aug_data: dataframe containing synthetic generated samples together with real samples
        after shuffling. 
        samples: dataframe containing ONLY synthetic generated samples. 
    """
    
    # Fit Synthetic Data Generator to controls
    model.fit(augmenting_params[2])

    # Generates "num" synthetic samples 
    ctrl_samples = model.sample(round(augmenting_params[4][size_index]/2), randomize_samples = False)

    # Fit Synthetic Data Generator to cases
    model.fit(augmenting_params[3])

    # Generates "num" synthetic samples 
    cases_samples = model.sample(round(augmenting_params[4][size_index]/2), randomize_samples = False)

    # Join all the data in one dataframe
    augmenting_params[2].reset_index(drop=True, inplace=True)
    augmenting_params[3].reset_index(drop=True, inplace=True)
    ctrl_samples.reset_index(drop=True, inplace=True)
    cases_samples.reset_index(drop=True, inplace=True)
    
    # Joint real control and cases
    original_data = pd.concat([augmenting_params[2], augmenting_params[3]])
    original_data.reset_index(drop=True, inplace=True)

    # Joint synthetic control and cases
    samples = pd.concat([ctrl_samples, cases_samples])
    samples.reset_index(drop=True, inplace=True)

    # Join all the data in one dataframe
    aug_data = pd.concat([original_data, samples])

    # Re-order instances (shuffle)
    aug_data = aug_data.sample(frac=1)
    
    # Store the dataset  
    if SAVE_DATASETS == True: 
        if not os.path.exists(r"./synthetic_datasets"):
            os.mkdir(r"./synthetic_datasets")
        os.chdir(r"./synthetic_datasets")
        filename = ("data_"+ augmenting_params[1] + "_sep_" + augmenting_params[0] + "_size"+"%d"+"_iter_"+"%d"+".sav") % (size_index, iter)
        pickle.dump(aug_data, open(filename, 'wb')) 

        # Return to previous directory 
        os.chdir(r"..") 
    
    return aug_data, samples

def data_aug_cond_after_split(model, augmenting_params : Tuple [str, str, pd.DataFrame, pd.DataFrame, List],
             conds : Tuple[Dict, Dict], size_index : int = 0, iter : int = 0, SAVE_DATASETS : bool = False) -> Tuple [pd.DataFrame, pd.DataFrame]:
    
    """Generates synthetic samples given a Synthetic Generator model after splitting the dataset
    into two sub-datasets containing only controls (e.g., Non-Diabetic patient) and cases 
    (e.g., Diabetic patients). Data generation is performed separately and the the two subdatasets
    are then joint. The number of generated samples is given by the "num" variable. This function has
    only been used with Gaussian Copula and CTGAN from SDV libraries. It shall work fine with other 
    generators or it shall not. For more information, please refer to https://sdv.dev/SDV/.

    Args:
    -----
        model: synthetic data generator model (so far only CTGAN or Gaussian Copula from https://sdv.dev/SDV/
        have been tested)
        augmenting_params: Tuple [augmentation_technique, balancing_tecnique, controls, cases, num_samples]. 
        First two elements are strings to properly handle and store the dataset. Next elements are the dataframes 
        containing the controls and cases datasets, respectively. The last element is a list with the diffrent 
        amounts of synthetic samples to be generated. Order must be as described to ensure proper functioning of this code.  
        conds: conditions to generate the dataset (e.g., set Glucose value over 100). Each dictionary contains
        a different condition. Order must be [CONDITION NEGATIVE, CONDITION POSITIVE] to a proper functioning
        of this code. 
        size_index: when more than one size is tested, indicates the index of it.
        iter: synthetic data generation iteration index 
        SAVE_DATASETS: when set to True, all generated datasets are stored using pickle in ./synthetic_datasets. When False, 
        dataset is generated without being stored.  

    Returns:
    --------
        aug_data: dataframe containing synthetic generated samples together with real samples
        after shuffling. 
        samples: dataframe containing ONLY synthetic generated samples. 
    """
    
    # Fit Synthetic Data Generator to controls
    model.fit(augmenting_params[2])

    # Creates control condition for all generated rows
    condition = Condition(conds[0], num_rows=(round(augmenting_params[4][size_index]/2)))
    
    # Generates "num" synthetic samples 
    ctrl_samples = model.sample_conditions([condition], 100, randomize_samples = False)

    # Fit Synthetic Data Generator to cases
    model.fit(augmenting_params[3])

    # Creates control condition for all generated rows
    condition = Condition(conds[1], num_rows=(round(augmenting_params[4][size_index]/2)))
    
    # Generates "num" synthetic samples 
    cases_samples = model.sample_conditions([condition], 100, randomize_samples = False)

    # Join all the data in one dataframe
    augmenting_params[2].reset_index(drop=True, inplace=True)
    augmenting_params[3].reset_index(drop=True, inplace=True)
    ctrl_samples.reset_index(drop=True, inplace=True)
    cases_samples.reset_index(drop=True, inplace=True)
    
    # Joint real control and cases
    original_data = pd.concat([augmenting_params[2], augmenting_params[3]])
    original_data.reset_index(drop=True, inplace=True)

    # Joint synthetic control and cases
    samples = pd.concat([ctrl_samples, cases_samples])
    samples.reset_index(drop=True, inplace=True)

    # Join all the data in one dataframe
    aug_data = pd.concat([original_data, samples])

    # Re-order instances (shuffle)
    aug_data = aug_data.sample(frac=1)
    
    # Store the dataset  
    if SAVE_DATASETS == True: 
        if not os.path.exists(r"./synthetic_datasets"):
            os.mkdir(r"./synthetic_datasets")
        os.chdir(r"./synthetic_datasets")
        filename = ("data_"+ augmenting_params[1] + "_sep_" + augmenting_params[0] + "_size"+"%d"+"_iter_"+"%d"+".sav") % (size_index, iter)
        pickle.dump(aug_data, open(filename, 'wb'))  
    
        # Return to previous directory 
        os.chdir("..") 
    
    return aug_data, samples

def basic_stats(X : pd.DataFrame, technique : str, file: str) -> np.array:
    """From a given dataset, mean, std., skewness and kurtosis are
    computed and saved in a .csv file, pointing out wich technique 
    has been used to generate the evaluated data. 
    a .csv file and . 

    Args:
    -----
        X: dataframe containing the synthetic dataset
        technique: name of the employed synthetic data generator
        file: .csv file name

    Returns:
    --------
        stats: array cotaining the statistical parameters of the dataset. 
    """
    # Empty numpy array to be filled with the statistical parameters
    stats = np.zeros((len(X.columns),4), float)
    
    # Write first line in file 
    file.write("%s\nmean,\tstd.,\tskewness,\tkurtosis\t\n" % technique)
    
    i = 0
    for feature in X :
        
        # Compute and store parameters 
        stats[i,0] = X[feature].mean()
        stats[i,1] = X[feature].std() 
        stats[i,2] = X[feature].skew()
        stats[i,3] = X[feature].kurtosis()
        file.write("%f,\t%f,\t%f,\t%f\n" % (stats[i,0],stats[i,1],stats[i,2],stats[i,3]))
        i = i+1

    return stats
 	
class Positive(Constraint):
    def __init__(self, column_name,handling_strategy='reject_sampling'):
        self._column_name = column_name
        super().__init__(handling_strategy=handling_strategy)

    def is_valid(self, table_data):
        """Say if values are positive."""
        column_data = table_data[self._column_name]
        positive = column_data >= 0 
        
        return positive

class Binary(Constraint):
    "Ensure model output is properly controlled"
    def __init__(self, column_name,handling_strategy='reject_sampling'):
        self._column_name = column_name
        super().__init__(handling_strategy=handling_strategy)

    def is_valid(self, table_data):
        """Say if values are equels to '0 or 1."""
        column_data = table_data[self._column_name]
        positive = column_data >= 0 
        less_than_one = column_data <= 1     
        return positive & less_than_one



	   
	   
	   


