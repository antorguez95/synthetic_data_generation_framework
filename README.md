# Synthetic Data Generation Framework

## What's in this repository?

This repository contains the code of our published work in the Journal of Biomedical and Health informatics: "Synthetic Patient Data Generation and 
Evaluation in Disease Prediction Using Small and Imbalanced Datasets". The main objective of this work was to demonstrate the feasibility of the employment of synthetic data to train Machine Learning models validated with real medical tabular data for classification tasks. This, without harming the statistical properties of the original data.
Obtained results demonstrates that, using [CTGAN](https://arxiv.org/abs/1907.00503) and a Gaussian Copula of the , [SDV library](https://sdv.dev/SDV/) classification performances can be perfectly maintained, and in some cases even improved. Further research must be done in this line, yet the results present in our work are promising. 

Please cite our paper #####PAPER if this framework somehow help you in your research and/or development work.

This framework is entirely developed in Python language. 

## Datasets Availability 

9 different databases has been used to test this framework. Most of them are publicly available. The rest, available under request to the authors. Aiming replicability of this expermients to further improvements, links to the databases (when possible) are provided below: 
1) MNCD
2) MNCD-Reduced (version with more patients and less features than MNCD)
3) Bangladesh 
4) Early Diabetes Mellitus 
5) Framingham Cardiovasvcular Diseases Risk Prediction
6) Heart Disease
7) Kidney Chronic Disease
8) Diabetes PIMA Database
9) South Africa Cardio

## How do I run these scripts?

In order to avoid code errors with libraries dependencies or misfunction, please install the needed libraries by running this line of code
in your conda prompt. 

  FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"
 
 where [`requirements.txt`](requirements.txt) contained the installed libraries. 


## Generated results

Due to their particularities, each database has its own script. The execution of each of them will generate am `EDA` and `results` folders that contain the initial Exploratory Data Analysis (EDA) and the results after data augmentation, respectively. Results are stored as figures or as `.pkl` files containing the numerical values of the metrics analysed. Please, refer to ######PAPER###### for further information regarding the studied metrics and obtained results. 

## Learn more 

For any other questions related with the code or the synthetic data framework ### PAPER itself, you can post an issue on this repository or contact me sending an email


