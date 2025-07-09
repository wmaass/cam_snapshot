from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import h2o
import pandas as pd
import matplotlib.pyplot as plt

# Initialize H2O
h2o.init()

# Application name
app_name = "HIS17-two-classes"
d_name = "hdma_1M.csv"

# Paths to train and test CSV files
path_data = './data/'
path_df_train = './results/' + app_name + '/ranked/train.csv'
path_df_test = './results/' + app_name + '/ranked/test.csv'
path_df_original = './results/'+app_name+'/ranked/df_original.csv'
path_model = './results/' + app_name + '/models/'
path_performance_images = './results/' + app_name + '/perf_images/'

target = "Total Costs"
leaky_features = 'Total Charges'
del_feature_names = ['Discharge Year', 'Birth Weight', 'Abortion Edit Indicator']
#    df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')
df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')

print(df.columns)

# Sample 100,000 rows
df = df.sample(n=200000, random_state=42)

# Remove only Total Charges to prevent data leakage
df = df.drop([leaky_features], axis=1)

# Convert 'Length of Stay' to numeric if it's not already
df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')

df.to_csv(path_df_original)

# Specify categorical features
cat_features = ['Hospital Service Area', 'Hospital County', 'Facility Name', 'Age Group',
                    'Zip Code', 'Gender', 'Race', 'Ethnicity', 'Type of Admission',
                    'Patient Disposition', 'CCS Diagnosis Description', 'CCS Procedure Description',
                    'APR DRG Description', 'APR MDC Code', 'APR MDC Description',
                    'APR Severity of Illness Code', 'APR Severity of Illness Description',
                    'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1',
                    'Payment Typology 2', 'Payment Typology 3',
                    'Emergency Department Indicator', 'Operating Certificate Number',
                    'Permanent Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'APR DRG Code']

# Identify numerical columns
num_features = [col for col in df.columns if col not in cat_features and col != target]

# Specify predictors and response
#predictors = num_features + [f'encoded_{feature}' for feature in cat_features]
predictors = num_features + cat_features

# delete del_feature_names from predictors
for feature in del_feature_names:
    if feature in predictors:
        predictors.remove(feature)

print(predictors)

response = target

# Read train and test datasets into pandas DataFrames
train = pd.read_csv(path_df_train)
test = pd.read_csv(path_df_test)

# Convert pandas DataFrames to H2OFrames
train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

# Define hyperparameter grid
hyper_params = {
    'ntrees': [100, 200, 300, 400],
    'max_depth': [1, 3, 8, 10, 12],
    'learn_rate': [0.01, 0.03, 0.05, 0.8, 0.1, 0.3, 0.5],
    'col_sample_rate': [0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
}

# Define search criteria
search_criteria = {
    'strategy': "RandomDiscrete",
    'max_models': 20,
    'seed': 1234
}

# Initialize the grid search
grid = H2OGridSearch(
    model=H2OGradientBoostingEstimator(seed=1234),
    hyper_params=hyper_params,
    search_criteria=search_criteria
)

# Train the model on the grid
grid.train(x=predictors, 
           y=response, 
           training_frame=train_h2o, 
           validation_frame=test_h2o)

# Display the grid results sorted by validation error
#print(grid.get_grid(sort_by='rmse', decreasing=False))

# Retrieve the grid results sorted by RMSE
sorted_grid = grid.get_grid(sort_by='rmse', decreasing=False)

# Store R², MAE, and hyperparameters for each model
metrics = []

# Iterate through each model in the grid
for model in sorted_grid.models:
    # Evaluate model performance on the test set
    performance = model.model_performance(test_data=test_h2o)
    
    # Get RMSE, R², and MAE
    rmse = performance.rmse()
    r2 = performance.r2()
    mae = performance.mae()
    
    # Extract hyperparameters
    col_sample_rate = model.params['col_sample_rate']['actual']
    learn_rate = model.params['learn_rate']['actual']
    max_depth = model.params['max_depth']['actual']
    ntrees = model.params['ntrees']['actual']
    
    # Append metrics and hyperparameters to the list
    metrics.append({
        'col_sample_rate': col_sample_rate,
        'learn_rate': learn_rate,
        'max_depth': max_depth,
        'ntrees': ntrees,
        #'model_id': model.model_id,
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    })

# Convert to DataFrame for better visualization
import pandas as pd
metrics_df = pd.DataFrame(metrics)

# Sort by RMSE or any other metric
metrics_df = metrics_df.sort_values(by='rmse', ascending=True)
print(metrics_df)

# Visualization of results
# 1. Scatter plot of RMSE vs R², colored by learning rate
plt.figure(figsize=(10, 6))
scatter = plt.scatter(metrics_df['r2'], metrics_df['rmse'], c=metrics_df['learn_rate'], cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Learning Rate')
plt.xlabel('R²')
plt.ylabel('RMSE')
plt.title('RMSE vs R² with Learning Rate Color Scale')
plt.grid(True)

plt.savefig(path_performance_images+'r2_rmse.png')
plt.show()

print(f"Image R2 vs. RMSE saved to: {path_performance_images}")

# 2. Line plot of RMSE by max_depth for various ntrees
plt.figure(figsize=(10, 6))
for ntrees in metrics_df['ntrees'].unique():
    subset = metrics_df[metrics_df['ntrees'] == ntrees]
    plt.plot(subset['max_depth'], subset['rmse'], label=f'ntrees={ntrees}', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.title('RMSE by Max Depth for Different ntrees')
plt.legend()
plt.grid(True)

plt.savefig(path_performance_images+'max_depth_rmse.png')
plt.show()

print(f"Image Max Depth vs RMSE saved to: {path_performance_images}")

# 3. Boxplot of R² by max_depth
plt.figure(figsize=(10, 6))
metrics_df.boxplot(column='r2', by='max_depth', grid=False)
plt.title('R² by Max Depth')
plt.suptitle('')  # Remove automatic matplotlib title
plt.xlabel('Max Depth')
plt.ylabel('R²')

plt.savefig(path_performance_images+'max_depth_r2.png')
plt.show()

print(f"Image Max Depth vs. R2 saved to: {path_performance_images}")

# Retrieve the best model
best_model = grid.get_grid(sort_by='rmse', decreasing=False).models[0]

# save best model
best_model_path = h2o.save_model(model=best_model, path=path_model, force=True)
print(f"Best model saved to: {best_model_path}")

# Evaluate the best model
perf = best_model.model_performance(test_data=test_h2o)
print(perf)

