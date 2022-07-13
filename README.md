# Synthetic Data Generation Framework

## What's in this repository?

This repository contains the code of our published work in the Journal of Biomedical and Health informatics: "Synthetic Patient Data Generation and 
Evaluation in Disease Prediction Using Small and Imbalanced Datasets"####PAPER. The main objective of this work was to demonstrate the feasibility of the employment of synthetic data to train Machine Learning models validated with real medical tabular data for classification tasks. This, without harming the statistical properties of the original data. With this goal, an in-depth analysys of the relationship between the amount of synthetic data samples, classification performance, and statistical similarity metrics was performed. 

Obtained results demonstrate that, using [CTGAN](https://arxiv.org/abs/1907.00503) and a Gaussian Copula available at the [SDV library](https://sdv.dev/SDV/), classification performances can be perfectly maintained, and even improved in some cases. Further research must be done in this line, yet the results present in our work are promising. 

`results` folder contained the most relevant results, most of them published in our work####PAPER. Further executions will overwrite the original results if neither the folder nor the file names are properly change within the code. `EDA` folders has not been included yet, since some errors arise when dealing with categorical variables. With `PIMA` and `SA-Cardio` databases Exploratory Data Analysis functions work becuase these datasets do not contain categorical variables. 

Please cite our paper #####PAPER if this framework somehow help you in your research and/or development work, or if you use this piece of code. 

## Datasets Availability 

8 different databases has been used to test this framework. Most of them are publicly available. The rest, available under request to the authors. Aiming replicability of this expermients to further improvements, links to the databases (or to the reference, when data is not freely available), are provided below: 

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
3. Enter the 'synthetic_data_generation' directory 
4. Create the environment by running:

    ```
    conda env create -f environment.yml -n SGD
    ```
   
 5. Activate the already installed envirionment by running: 

    ```
    conda activate SGD
    ```

where [`environment.yml`](environment.yml) contained the name of the installed environment, the installed libraries and the channel used to download such libraries.

## How do I run these scripts?

To execute the whole experiment, with the default settings, this is the line of code you must type in a Python terminal. Changes will be introduced 
to input the parameters from the terminal,
    
    cd DATASET_FOLDER
    python DATASET_NAME_main.py
 
where `DATASET_FOLDER` must be replaced by the folder correspondant to the dataset (e.g., `PIMA`) and `DATASET_NAME` must be replaced by one of the 8 used databases (e.g., `PIMA`). 

Once the results have been already generated, one can visualize some figures after loading the obtained results without the need of re-executing everything. The choice of the dataset to be analyzed must be done inside the  `gen_and_save.py` for now, as indicated in the comments of such file. Changes will be introduced to input the parameters from the terminal.
From the main folder, execute this:
 
    python gen_and_save_figs.py
  
## Generated results

Due to their particularities, each database has its own script. The execution of each of them will generate am `EDA` and `results` folders that contain the initial Exploratory Data Analysis (EDA) and the results after data augmentation, respectively. Results are stored as figures or as `.pkl` files containing the numerical values of the metrics analysed. Please, refer to ######PAPER###### for further information regarding the studied metrics and obtained results. 

## Learn more 

For any other questions related with the code or the synthetic data framework ### PAPER itself, you can post an issue on this repository or contact me sending an email


