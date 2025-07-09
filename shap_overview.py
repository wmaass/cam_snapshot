import pandas as pd
import shap
import matplotlib.pyplot as plt
import os


path_data = '~/Dropbox/CAData/'
app_name = "NYHealthSPARC"

# d_name = "xaa.csv"
# d_name_short = "LOAN"
# df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

# predictors_all = ['applicant_sex', 'purchaser_type', 'applicant_age','loan_term', 'origination_charges', 'derived_loan_product_type', 'property_value',
#                   'loan_amount',  'total_loan_costs', 'rate_spread', 'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose', 
#                   'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose']

# df = df[predictors_all]

# # Load the SHAP values from CSV
# #shap_file_path = './results/HIS17-random/sh_dfdecision.csv'
# shap_file_path = './results/LOAN/sh_dfdecision.csv'
# shap_values_df = pd.read_csv(shap_file_path, low_memory=False)

#-----

# Create output directory if it doesn't exist
figure_path = './figures/'+app_name
os.makedirs(figure_path, exist_ok=True)

d_name = "hdma_1M.csv"
d_name_short = "HIS17"
df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

# Create output directory if it doesn't exist
figure_path = './figures/' + app_name
os.makedirs(figure_path, exist_ok=True)


predictors_all = ["Zip Code", "Permanent Facility Id", "Hospital County", "Race", "Ethnicity", "Age Group", "Length of Stay", "Hospital Service Area", 
                             "Facility Name", "Payment Typology 1", "APR MDC Description", "Emergency Department Indicator", "APR Severity of Illness Description", 
                             "APR Risk of Mortality", "CCS Diagnosis Description", "Gender", "Patient Disposition", "CCS Procedure Code",
                             "CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Payment Typology 2", "APR DRG Description", 
                             "APR DRG Code", "Payment Typology 3", "Type of Admission", "APR Severity of Illness Code", "Operating Certificate Number", 
                             "APR Medical Surgical Description"]

df = df[predictors_all]

# Load the SHAP values from CSV
shap_file_path = './results/'+ app_name +'/shap_mlsmall.csv'
shap_values_df = pd.read_csv(shap_file_path, low_memory=False)

# drop first column
shap_values_df = shap_values_df.drop(shap_values_df.columns[0], axis=1)

# ----------------------------------------------------------------

# Assuming the CSV contains feature names as columns and SHAP values as rows
# Separate the features and SHAP values
features = shap_values_df.columns
shap_values = shap_values_df.values

# This needs to be calculated or provided separately based on your model
base_values = shap_values.mean(axis=0)

# Generate a waterfall plot for the first observation
shap.initjs()  # Ensure that the plots will be rendered

# Creating an Explanation object for the first three observations

for i in range(1):

    explainer = shap.Explanation(values=shap_values[i], 
                                base_values=base_values[i], 
                                data=None, 
                                feature_names=features)

    # Set up the figure size and margins for the waterfall plot
    plt.figure(figsize=(12, 8))  # Increase the figure size
    plt.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.1)  # Adjust the margins

    # Generate the waterfall plot
    shap.waterfall_plot(explainer)

# Compute the mean SHAP values for each feature
mean_shap_values = shap_values.mean(axis=0)

# Create a bar plot to visualize the mean SHAP values
plt.figure(figsize=(10, 8))
plt.barh(features, mean_shap_values, color='blue')
plt.xlabel('mean(SHAP value)')
plt.title('Mean SHAP values for each feature')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest values at the top
plt.show()

# Plot the SHAP summary plot
plt.figure(figsize=(12, 8))  # Increase the figure size
plt.subplots_adjust(left=0.2, right=0.6, top=0.9, bottom=0.1)  # Adjust the margins
shap.summary_plot(shap_values, features=shap_values_df, feature_names=features, plot_type="bar")

# Optional: To plot the detailed summary plot
shap.summary_plot(shap_values, features=shap_values_df, feature_names=features, show=False)

# save shap.summary_plot
plt.savefig(os.path.join(figure_path, f"SHAP_summaryplot.png"))

