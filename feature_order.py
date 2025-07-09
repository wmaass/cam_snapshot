import pandas as pd
import numpy as np
import os

import calign21
from calign21 import * # that's the problem for reloading issues!!! use calign21.function instead
import importlib
importlib.reload(calign21)

from sklearn.preprocessing import LabelEncoder

#import tensorflow as tf
import GPUtil

# test GPU
GPUs = GPUtil.getGPUs()
print(GPUs)

if GPUs:
    path_data = '../../data/'
else:
    path_data = '~/Dropbox/CAData/'
    #path_data = './CAData/'

app_name = "NYHealthSPARC" 
# Data Access - Statewide Planning and Research Cooperative System (SPARCS)
# https://health.data.ny.gov/dataset/Hospital-Inpatient-Discharges-SPARCS-De-Identified/22g3-z7e7/about_data


path_data = '~/Dropbox/CAData/'
d_name = "hdma_10k2.csv"
d_name_short = "HIS17"
df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

predictors_all = ["Zip Code", "Permanent Facility Id", "Hospital County", "Race", "Ethnicity", "Age Group", "Length of Stay", "Hospital Service Area", 
                             "Facility Name", "Payment Typology 1", "APR MDC Description", "Emergency Department Indicator", "APR Severity of Illness Description", 
                             "APR Risk of Mortality", "CCS Diagnosis Description", "Gender", "Patient Disposition", "CCS Procedure Code",
                             "CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Payment Typology 2", "APR DRG Description", 
                             "APR DRG Code", "Payment Typology 3", "Type of Admission", "APR Severity of Illness Code", "Operating Certificate Number", 
                             "APR Medical Surgical Description"]

# Create output directory if it doesn't exist
figure_path = './figures/'+app_name
os.makedirs(figure_path, exist_ok=True)

def calculate_average_total_costs(df, column_name):
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    print(df[column_name].head())

    # Convert 'Total Costs' to numeric, coercing errors
    df['Total Costs'] = pd.to_numeric(df['Total Costs'], errors='coerce')

    # Encode the specified column with numerical values
    le = LabelEncoder()
    df[column_name + '_Encoded'] = le.fit_transform(df[column_name])

    # Convert column_name + '_Encoded' to numeric, coercing errors
    df[column_name + '_Encoded'] = pd.to_numeric(df[column_name + '_Encoded'], errors='coerce')

    # Create a dictionary for the original values and their encodings
    value_encoding_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Group by the encoded column and calculate the average of 'Total Costs'
    average_costs = df.groupby(column_name + '_Encoded')['Total Costs'].mean().reset_index()

    # replace NaN values in average_costs with 0 values

    average_costs['Total Costs'].fillna(0, inplace=True)
    
    # Map the encoded values back to the original values
    average_costs[column_name] = average_costs[column_name + '_Encoded'].map(lambda x: le.inverse_transform([x])[0])

    # sort average_costs by column 'Total Costs'
    average_costs.sort_values(by='Total Costs', ascending=True, inplace=True)

    # Drop the encoded column
    average_costs.drop(columns=[column_name + '_Encoded'], inplace=True)

    # reset index
    average_costs.reset_index(drop=True, inplace=True)

    # Rename the columns
    average_costs.columns = ['Average Total Costs', column_name]

    print(average_costs)

    # Save the average costs DataFrame to a CSV file
    average_costs.to_csv(os.path.join(figure_path, f"{column_name}_average_costs.csv"), index=True)

    # Create a mapping from original values to their order in the average_costs dataframe
    value_to_order_dict = {value: idx for idx, value in enumerate(average_costs[column_name])}
    
    # Use this mapping to create the new column in the original dataframe
    df[column_name + '_order'] = df[column_name].map(value_to_order_dict)

    df = df.drop(columns=[column_name])

    # rename column in df
    df.rename(columns={column_name + '_order': column_name}, inplace=True)

    # Drop the encoded column
    df = df.drop(columns=[column_name + '_Encoded'])

    return df

column_name_list = ['Patient Disposition']  # Replace with the actual column name you want to analyze

for cn in column_name_list:
    df = calculate_average_total_costs(df, cn)

print(df)
