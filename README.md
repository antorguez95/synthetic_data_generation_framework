# Synthetic Data Generation Framework

## What's in this repository?

This repository contains the code of our published work in the [IEEE Journal of Biomedical and Health Informatics](https://www.embs.org/jbhi/): [*"Synthetic Patient Data Generation and 
Evaluation in Disease Prediction Using Small and Imbalanced Datasets"*](https://ieeexplore.ieee.org/document/9851514). The main objective of this work was to demonstrate the feasibility of the employment of synthetic data to train Machine Learning models validated with real medical tabular data for classification tasks. This, without harming the statistical properties of the original data. With this goal, an in-depth analysis of the relationship between the amount of synthetic data samples, classification performance, and statistical similarity metrics was performed. 

There are 8 folders, one for each `database`. Inside each folder there are two files: `database_utils.py` and `database_main.py` (being `database` the correspondant name for each database). The former contains constants and custom functions developed uniquely for that database. The latter contains the most important part of this work; the script with the framework itself. One script has been developed for each database due to the heterogeneity and particularities of all databases. The arguments/parameters of every script are on the top of them. Notice that, with the default parameters (`bal_iterations = 100`, `aug_iterations = 10`), and the current grid of Machine Learning models parameters (see `svm_params`, `rf_params`, `xgb_params`, `knn_params` variables) execution time can last from around 6 hours to nearly a day, depending on the database. Reduction of iterations and/or grid parameters will reduce the execution time.

`results` folders contain the most relevant results, most of them published in [our work](https://ieeexplore.ieee.org/document/9851514). Further executions of this code will overwrite the original results if neither the folder nor the file names are properly changed within the code. `EDA` folders has not been included yet, even they are generated when executing this code, since some errors arise when dealing with categorical variables. With `PIMA` and `SACardio` databases Exploratory Data Analysis (EDA) functions work because these datasets do not contain categorical variables. 

Obtained results demonstrate that, using [CTGAN](https://arxiv.org/abs/1907.00503) and a Gaussian Copula available at the [SDV library](https://sdv.dev/SDV/), classification performances can be perfectly maintained, and even improved in some cases. Further research must be done in this line, yet the results presented in [our work](https://ieeexplore.ieee.org/document/9851514) are promising. 

Please cite [our paper](https://ieeexplore.ieee.org/document/9851514) if this framework somehow helped you in your research and/or development work, or if you used this piece of code: 

*A. J. Rodriguez-Almeida et al., "Synthetic Patient Data Generation and Evaluation in Disease Prediction Using Small and Imbalanced Datasets," in IEEE Journal of Biomedical and Health Informatics, 2022, doi: 10.1109/JBHI.2022.3196697.*

## Datasets Availability 

8 different databases has been used to test this framework. Most of them (6) are publicly available. The rest (2) are available under request to the authors. Aiming replicability of this expermient, links to the databases (or to the reference, when data is not freely available), are provided below: 

1) [MNCD](https://pubmed.ncbi.nlm.nih.gov/33361594/)
2) MNCD-Reduced (version with more patients and less features than MNCD)
3) [Bangladesh](https://www.kaggle.com/datasets/sabbir1996/dataset-of-diabetes-type1)
4) [Early Diabetes Mellitus](https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset) 
5) [Heart Disease](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
6) [Kidney Chronic Disease](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
7) [Diabetes PIMA Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
8) [South Africa Cardio](https://www.kaggle.com/datasets/yassinehamdaoui1/cardiovascular-disease)

## Set up before running this code
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Enter the `synthetic_data_generation` directory 
4. Create the environment with the proper Python version by running:

    ```
    conda create -n SDG python=3.8.13
    ```
   
 5. Activate the already installed envirionment by running: 

    ```
    conda activate SGD
    ```
 6. Install the required packages. Solutions for this using `requirements.txt` or `environments.yml` has been tested but `sdv` package show conflicts due to cross-dependencies with other libraries. So, packages sould be installed manually. Notice that the installed versions are not the most recent ones, but the ones that have been employed to develop and test this framework. In your terminal run (one line by one): 
    
    ```
    conda install scikit-learn=1.0.2
    conda install pandas=1.1.3
    conda install numpy=1.21.5
    conda install -c conda-forge imbalanced-learn=0.7.0
    conda install matplotlib=3.5.2
    conda install -c pytorch -c conda-forge sdv=0.14.1
    conda install openpyxl=3.0.9
    ``` 
 
 8. Download the databases and set the `DATASET_PATH` in all `datasetname_main.py` files according to your own path. Check also that `filename` variable      contains the actual file name of the database. Finally, set the `DICT_PATH` variable in all `datasetname_utils.py` to store the dictionaries that contain the results properly. 

where [`environment.yml`](environment.yml) contained the name of the installed environment, the installed libraries and the channel used to download such libraries, and `datasetname` corresponds to the abovementioned datasets names. 

## How do I run these scripts?

To execute the whole experiment, with the default settings, these are the lines of code you must type in your Python terminal. Changes in the code will be introduced to input the parameters from the terminal. From the `synthetic_data_generation` folder: 
    
    cd DATASET_FOLDER
    python DATASET_NAME_main.py
 
where `DATASET_FOLDER` must be replaced by the folder correspondant to the dataset (e.g., `PIMA`) and `DATASET_NAME` must be replaced by one of the 8 used databases (e.g., `PIMA`). Notice that this lines must be executed one time per database to obtain all the results.  

Once the results have been already generated, one can visualize some figures after loading the obtained results without the need of re-executing everything. If you do not have `Latex` installed in your PC, please do one of the following things: 
    
- Install it. 
- Comment every line that contains  `plt.style.use(['science','ieee'])`  within the code. 

The choice of the dataset to be analyzed must be done inside the `gen_and_save.py` file for now, as indicated in the comments of such file. Specifically, `STUDIED_DATABASE` variable must be properly set. Changes will be introduced to input the parameters from the terminal. Afterwards, execute this line from the main folder:
 
    python gen_and_save_figs.py
  
## Generated results

As previously outlined, due to their particularities, each database has its own script. The execution of each of them will generate an `EDA` and `results` folders that contain the initial EDA and the results after data augmentation, respectively. Results are stored as figures, as `.pkl` files and/or as `.txt` files containing the numerical values of the metrics analysed. Please, refer to [our paper](https://ieeexplore.ieee.org/document/9851514) for further information regarding the studied metrics and obtained results. 

## Learn more 

For any other questions related with the code or the [synthetic data framework](https://ieeexplore.ieee.org/document/9851514) itself, you can post an issue on this repository or contact me via email.


