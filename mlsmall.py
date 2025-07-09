import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

app_name = 'NYHealthSPARC'

path_results = './results/'+ app_name + '/'

if app_name == 'NYHealthSPARC':
    # Load the data
    path_data = '~/Dropbox/CAData/'
    d_name = "hdma_1M.csv"
    df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')

    # Define the target and predictors
    target = "Total Costs"
    predictors_all = ["Zip Code", "Permanent Facility Id", "Hospital County", "Race", "Ethnicity", "Age Group", "Length of Stay", 
                    "Hospital Service Area", "Facility Name", "Payment Typology 1", "APR MDC Description", 
                    "Emergency Department Indicator", "APR Severity of Illness Description", "APR Risk of Mortality", 
                    "CCS Diagnosis Description", "Gender", "Patient Disposition", "CCS Procedure Code", 
                    "CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Payment Typology 2", 
                    "APR DRG Description", "APR DRG Code", "Payment Typology 3", "Type of Admission", 
                    "APR Severity of Illness Code", "Operating Certificate Number", "APR Medical Surgical Description"]

    # Ensure 'total_costs' column exists in the DataFrame (this line assumes 'total_costs' is present)
    if 'Total Costs' not in df.columns:
        raise ValueError("'Total Costs' column not found in the dataset")

# Preprocess the categorical data using Label Encoding
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))

# Split the data into train and test sets
X = df[predictors_all]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost parameters
# params = {
#     'objective': 'reg:squarederror',
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'eval_metric': 'rmse'  # Specify evaluation metric
# }

params = {'colsample_bytree': 0.8, 
          'learning_rate': 0.3, 
          'max_depth': 6, 
          'min_child_weight': 3, 
          'n_estimators': 300, 
          'subsample': 1.0}

# Train the XGBoost model with evaluation
evals_result = {}
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval'), (dtrain, 'train')],
                  evals_result=evals_result, early_stopping_rounds=10, verbose_eval=True)

# Save the model in JSON format to avoid the deprecation warning
model.save_model('xgboost_model.json')

# Predict and calculate performance metrics
y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualize the loss function over epochs
epochs = len(evals_result['train']['rmse'])
x_axis = range(0, epochs)

# Plot RMSE
fig, ax = plt.subplots()
ax.plot(x_axis, evals_result['train']['rmse'], label='Train')
ax.plot(x_axis, evals_result['eval']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()

# Generate SHAP values for the test set
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test, check_additivity=False)

sh_df = pd.DataFrame(shap_values.values)

# replace nan with 0
sh_df.fillna(0, inplace=True)

sh_df.columns = predictors_all

# save sh_df to csv file with name sh_df plus c 
sh_df.to_csv(path_results + 'shap_mlsmall.csv')



# Visualize the SHAP values with a summary plot
shap.summary_plot(shap_values, X_test)

# Visualize the SHAP values with a waterfall plot for the first instance
shap.waterfall_plot(shap_values[0])
