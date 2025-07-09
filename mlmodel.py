import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

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

# Ensure 'Total Costs' column exists in the DataFrame
if target not in df.columns:
    raise ValueError(f"'{target}' column not found in the dataset")

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

# Custom transformer to preserve feature names
class FeatureNamePreserver(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)

# Define a preprocessing and modeling pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_features)
    ],
    remainder='passthrough'
)

# Update XGBRegressor constructor with early_stopping_rounds and eval_metric
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', early_stopping_rounds=10)

pipeline = Pipeline(steps=[
    ('preprocessor', FeatureNamePreserver(preprocessor)),
    ('scaler', FeatureNamePreserver(StandardScaler())),
    ('model', xgb_model)
])

# param_grid = {
#     'model__max_depth': [3, 4, 5, 6, 7, 8],
#     'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'model__n_estimators': [100, 200, 300, 500],
#     'model__subsample': [0.6, 0.8, 1.0],
#     'model__colsample_bytree': [0.6, 0.8, 1.0],
# }

param_grid = {
    'model__max_depth': [3, 4, 5, 6, 7, 8],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__n_estimators': [100, 200, 300, 500],
    'model__subsample': [1.0],
    'model__colsample_bytree': [1.0],
}

# Custom scorer for GridSearchCV
def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return r2_score(y, y_pred)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring=custom_scorer, verbose=0, n_jobs=-1)

# Fit the model using grid search
grid_search.fit(X_train, y_train, model__eval_set=[(X_test, y_test)])

print("Best parameters found: ", grid_search.best_params_)
print("Best R^2 score: ", grid_search.best_score_)

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Predict and calculate performance metrics
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Extract the best XGB model from the pipeline
best_xgb_model = best_model.named_steps['model']

# Visualize the loss function over epochs
results = best_xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# Plot RMSE
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Test')

# Check if training evaluation is available
if 'validation_1' in results:
    ax.plot(x_axis, results['validation_1']['rmse'], label='Train')

ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()