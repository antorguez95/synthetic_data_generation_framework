# Synthetic Data Generation Framework

## What's in this repository?

This repository contains the code of our published work in the Journal of Biomedical and Health informatics: "Synthetic Patient Data Generation and 
Evaluation in Disease Prediction Using Small and Imbalanced Datasets". Please cite this paper if this framework somehow help you in your research and/or development work.

This framework is entirely developed in Python language. In 

## Datasets Availability 

9 different databases has been used to test this framework. Most of them are publicly available. The rest, available under request to the authors. Links to the databases are provided below: 
1) Alzh
2) Alz
3) Bangladesh 
4) Early Diabetes Mellitus 
5) Framingham Cardiovasvcular Diseases Risk Prediction
6) Heart Disease
7) Kidney Chronic Disease
8) Diabetes PIMA Database
9) SA-Cardio

## How do I run these scripts?

In order to avoid code errors with libraries dependencies or misfunction, please install the needed libraries by running this line of code
in your conda prompt. 

  FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"
 
 where [`requirements.txt`](requirements.txt) contained the installed libraries. 


## Generated results

Each database has its correspondant script. The execution of each of them will generate a `results` and  `EDA` folders that contain the results after data augmentation and an initial Exploratory Data Analysis (EDA), respectively. Results are stored as figures or as `.pkl` files containing the numerical values of the metrics analysed. Please, refer to ######PAPER###### for further information. 

## Learn more 

For any other questions related with the code you can post an issue on this repository or contact us sending an email


