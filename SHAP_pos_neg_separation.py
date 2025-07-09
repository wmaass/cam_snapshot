import pandas as pd
import numpy as np
import os

app_name = "NYHealthSPARC" 
cluster_num = 4
#cluster_type = 'kmeans'
cluster_type = 'llm'

# Data Access - Statewide Planning and Research Cooperative System (SPARCS)
# https://health.data.ny.gov/dataset/Hospital-Inpatient-Discharges-SPARCS-De-Identified/22g3-z7e7/about_data

# Attempt to read the CSV file from the specified path
#file_path = './results/HIS17-random/sh_dfdecision.csv'
#file_path = './results/'+app_name+'/shap_mlsmall.csv'

file_path = './results/'+app_name+'/shap_values_df.csv'

# Create output directory if it doesn't exist
figure_path = './figures/'+app_name
os.makedirs(figure_path, exist_ok=True)

# Create output directory if it doesn't exist
dist_path = './figures/'+ app_name + '/distributions'
os.makedirs(dist_path, exist_ok=True)

# Read SHAP values from file
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # drop column 'BiasTerm'
    df = df.drop('BiasTerm', axis=1)

    print(df.head())
else:
    print("File not found")

# Calculate the mean of each column
column_means = df.mean()

# Separate columns into df_pos and df_neg based on the sign of their means
df_pos = df.loc[:, column_means > 0]
df_neg = df.loc[:, column_means < 0]

# Display the resulting DataFrames
print("Columns with Positive Means:")
print(df_pos.head())

print("\nColumns with Negative Means:")
print(df_neg.head())

df_pos.to_csv('./results/'+app_name+'/shap_values_df_pos.csv', index=False)
df_neg.to_csv('./results/'+app_name+'/shap_values_df_neg.csv', index=False)
