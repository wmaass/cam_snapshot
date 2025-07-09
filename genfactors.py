import h2o
from h2o.automl import H2OAutoML
from h2o.explanation import explain
import matplotlib.pyplot as plt

app_name = "HIS17-two-classes" 

path_data = './data/'
path_rank_encodings = './results/'+app_name+'/ranked/rank_encoded.csv'


gbm_model_path = './results/'+app_name+'/ranked/GBM_model_python_1727883873584_1'

# Initialize and load data
h2o.init()

# read data from path_rank_encodings
data = h2o.import_file(path_rank_encodings)

# Drop the first column from the data
data = data[:, 1:]

# Get column names and store in a variable
columns = data.columns

print(data[:, :3].head(5))

# Train Gradient Boosting Model
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# load h2o gbm model
model = h2o.load_model(gbm_model_path)

# Feature importance
model.varimp_plot()

# Partial dependence
#model.partial_plot(data, cols=columns, nbins=50)

# SHAP values
shap_values = model.predict_contributions(data)
shap_summary = shap_values.as_data_frame()

# Plot SHAP summary
import shap
shap.summary_plot(shap_summary.values, data.as_data_frame())
