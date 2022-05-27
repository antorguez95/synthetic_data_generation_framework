import matplotlib.pyplot as plt
import os 
import pandas as pd
import seaborn as sn

def eda(data : pd. DataFrame, X : pd.DataFrame , Y : pd.DataFrame, dataset_name : str, folder : str = r"./EDA") :
     """Performs Exploratory Data Analysis (EDA) when a dataset if given:
     General information, dataset dimensions, amount of missing data, control/cases ratio, 
     histograms, boxplots and Pearson's correlation matrix are computed and stored. 

     Args:
     -----
          data: the whole dataset with features and target variable
          X: features of the dataset
          Y: target variable of the dataset
          dataset_name: name of the dataset to properly store the files 
          folder: folder to save the generated files. Defaults to r"./EDA"

     Returns:
     --------
          None 
     """
     # Save current working directory to come back later 
     cwd = os.getcwd()

     # Create folder to store EDA analysis files if not existing
     if not os.path.exists(folder):
          os.mkdir(folder)

     # Go to /EDA folder
     os.chdir(folder)

     # Print summarized information of the dataset
     print("\nDataset dimensions: %d x %d\n" % (data.shape[0], data.shape[1]))
     data.info(null_counts = True)

     # Calculate and save the Control/Cases ratio
     plt.figure()
     plt.bar([0,0.15], Y.value_counts(), width = 0.1, align = "center", tick_label = ["control","cases"])     
     for i, v in enumerate(Y.value_counts()):
          plt.text(i/7, 505, str(v), color='k', fontweight='bold')
     name = dataset_name + '_ctrl_cases'
     plt.savefig(name, dpi=600)
     print(Y.value_counts()*100/len(Y))

     # Generate and save histograms of all features
     X.hist(edgecolor = "black",column=None, by=None, grid=False, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=None, layout=None, bins=50, backend=None, legend=False)
     name = dataset_name+"_all_histograms"
     plt.savefig(name, dpi=300)

     # Genarate and save boxplots of all features 
     for feature in X: 
          plt.figure() 
          X.boxplot(feature, grid = False)
          name = "boxplot_" + feature
          plt.savefig(name, dpi=300)

     # Generate and save Pearson's correlation matrix
     plt.figure()
     corr_matrix = X.corr()
     sn.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap="vlag" ,annot=True)
     # Save figure 
     name = dataset_name + "_corr_matrix" 
     plt.savefig(name, dpi=300)

     # For categorical variables, check if there is any category missing
     for col in data :
          vals = data[col].nunique()
          print("Values of column %s: %s" % (col, vals))

     # Back to working directory 
     os.chdir(cwd)
      