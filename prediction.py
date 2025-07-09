import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import calign21
from calign21 import * # that's the problem for reloading issues!!! use calign21.function instead
import importlib
importlib.reload(calign21)

path_data = '~/Dropbox/CAData/'
d_name = "hdma_1M.csv"

# Specify dtype for columns with mixed types or set low_memory=False to avoid DtypeWarning
data = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1', low_memory=False)
y = data['Total Costs']

predictors_all = ["Zip Code", "Permanent Facility Id", "Hospital County", "Race", "Ethnicity", "Age Group", "Length of Stay", "Hospital Service Area", 
                             "Facility Name", "Payment Typology 1", "APR MDC Description", "Emergency Department Indicator", "APR Severity of Illness Description", 
                             "APR Risk of Mortality", "CCS Diagnosis Description", "Gender", "Patient Disposition", "CCS Procedure Code",
                             "CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Payment Typology 2", "APR DRG Description", 
                             "APR DRG Code", "Payment Typology 3", "Type of Admission", "APR Severity of Illness Code", "Operating Certificate Number", 
                             "APR Medical Surgical Description"]

features_randomize = []

X = pre_processing(data, predictors_all, features_randomize)

# Step 1: Separate features and target variable
# X = data.drop('Total Costs', axis=1)
# y = data['Total Costs']

# Step 2: Convert all columns to strings to ensure uniform data types
#X = X.astype(str)

# Step 3: Handle missing values
X = X.fillna('missing')

# Step 4: One-hot encode the categorical features
#categorical_columns = X.columns

# Initialize OneHotEncoder with updated parameter
#encoder = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform the categorical features
#X_encoded = encoder.fit_transform(X[categorical_columns])

# Convert encoded features back to a DataFrame
#X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train the XGBoost model for regression
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Optional: If you want to see the feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print(feature_importance.head(10))  # Print top 10 important features
