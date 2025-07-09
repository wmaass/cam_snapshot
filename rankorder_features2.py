import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import io
import warnings
import GPUtil
import os
import shap
import time

from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize H2O cluster
h2o.init(verbose=True)

h2o.cluster_info()
h2o.log_and_echo("Starting GBM Grid Search")

try:
    app_name = "HIS17-two-classes" 

    # Test GPU
    GPUs = GPUtil.getGPUs()
    print(GPUs)

    # print current path
    print(os.getcwd())

    if GPUs:
        path_data = '../../data/'
    else:
#        path_data = '~/Dropbox/CAData/'
        path_data = './data/'

    d_name = "hdma_1M.csv"
    d_name_short = "HIS17"

    path_ranked = './results/'+app_name+'/ranked/'

    # if path_ranked does not exist then create
    if not os.path.exists(path_ranked):
        os.makedirs(path_ranked)

    path_df_train = './results/'+app_name+'/ranked/train.csv'
    path_df_test = './results/'+app_name+'/ranked/test.csv'
    path_df_original = './results/'+app_name+'/ranked/df_original.csv'
    path_rank_encodings = './results/'+app_name+'/ranked/rank_encoded.csv'
    path_combined_ranking = './results/'+app_name+'/ranked/combined_ranking.csv'
    path_combined_ranking_data = './results/'+app_name+'/ranked/combined_ranking_data.csv'
    path_shap_values = './results/'+app_name+'/ranked/shap_values_df.csv'

    target = "Total Costs"
    leaky_features = 'Total Charges'
    del_feature_names = ['Discharge Year', 'Birth Weight', 'Abortion Edit Indicator']
#    df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')
    df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')

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
                        'APR Severity of Illness Code', 
                        #'APR Severity of Illness Description',
                        'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1',
                        'Payment Typology 2', 'Payment Typology 3',
                        'Emergency Department Indicator', 'Operating Certificate Number',
                        'Permanent Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'APR DRG Code']

    # Identify numerical columns
    num_features = [col for col in df.columns if col not in cat_features and col != target]

    print("Categorical features:", cat_features)
    print("Numerical features:", num_features)

    # Function to rank categories based on median cost
    def rank_categories(df, cat_features, del_feature_names, target_name=target):
        encoding_dict = {}
        for feature in cat_features:
            if feature in df.columns and df[feature].nunique() > 1:
                try:
                    median_costs = df.groupby(feature)[target_name].median().sort_values(ascending=True)
                    rankings = {cat: rank for rank, cat in enumerate(median_costs.index, 1)}
                    encoding_dict[feature] = rankings
                except Exception as e:
                    print(f"Error encoding feature {feature}: {str(e)}")
        return encoding_dict

    # Apply ranking encoding
    encoding_dict = rank_categories(df, cat_features, del_feature_names)
    for feature in cat_features:
        new_feature_name = f'encoded_{feature}'
        df[new_feature_name] = df[feature].map(encoding_dict[feature])
        df[new_feature_name] = df[new_feature_name].fillna(0)  # Handle unseen categories

    # from df store features predictors and response in df_rankings
    df_rankings = df.copy()
    df_rankings = df_rankings.drop(cat_features, axis=1)

    print(df_rankings['encoded_APR Severity of Illness Description'].head())

    num_features =[x for x in num_features if x not in del_feature_names]
    print("Numerical features:", num_features)

    df_rankings = df_rankings.drop(del_feature_names, axis=1)  # Drop 'Total Costs' column

    # delete 'encoded_' from all column names in df_rankings
    for feature in cat_features:
        df_rankings.rename(columns={f'encoded_{feature}': feature}, inplace=True)

    #df.rankings.to_csv()

    df_rankings.to_csv(path_rank_encodings, index=True)

    # drop cat_features from df
    #df = df.drop(cat_features, axis=1)

    # Convert to H2O frame
    #hf = h2o.H2OFrame(df)

    # Split the data
    # train, test = hf.split_frame(ratios=[0.8], seed=1234)

    # Split the dataframe, while keeping the index
    train, test = train_test_split(df_rankings, test_size=0.2, random_state=42)

    # Save train and test datasets to CSV while preserving the index
    train.to_csv(path_df_train, index=True)
    test.to_csv(path_df_test, index=True)

    # Convert train and test datasets to H2O frames
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)

    # Convert the test H2O frame to a pandas dataframe
    df_pros = h2o.as_list(test_h2o)

    # Select the first 10 rows of the test dataset
    df_pros = df_pros.iloc[:10]

    # Save the first 10 rows of the test dataset to a CSV file
    df_pros.to_csv('df_test_h20.csv', index=False)

    # Specify predictors and response
    # Combine numerical and categorical feature names for predictors
    predictors = num_features + cat_features

    # Debugging: Print the list of predictors
    print("Predictors:", predictors)

    # Set the target (response) variable
    response = target


    # # Train GBM model
    # gbm = H2OGradientBoostingEstimator(ntrees=400, max_depth=4, learn_rate=0.1, seed=1234)
    # gbm.train(x=predictors, y=response, training_frame=train_h2o, validation_frame=test_h2o)

    # # Save the GBM model to disk
    # model_path = h2o.save_model(model=gbm, path=path_ranked, force=True)

    # # Print the path to the saved model
    # print(f"Model saved to: {model_path}")

    # # Model performance
    # print("Model Performance:")
    # print(gbm.model_performance(test_h2o))

    ########################################

    # Define the Gradient Boosting Machine (GBM) model
    # gbm = H2OGradientBoostingEstimator(
    #     seed=1234,                              # Ensures reproducibility
    #     nfolds=5,                               # Number of folds for cross-validation
    #     keep_cross_validation_models=True,      # Retain cross-validation models for analysis
    #     keep_cross_validation_predictions=True, # Retain cross-validation predictions
    #     keep_cross_validation_fold_assignment=True # Retain CV fold assignment for diagnostics
    # )

    # # Define the hyperparameter grid for tuning
    # hyper_params = {
    #     "ntrees": [300, 500, 700],          # Number of trees
    #     "max_depth": [12, 13, 15],               # Maximum tree depth
    #     "learn_rate": [0.01, 0.02, 0.05],    # Learning rate (shrinkage)
    # }

    # # Create a grid search object
    # grid = H2OGridSearch(
    #     model=gbm,
    #     hyper_params=hyper_params,
    #     grid_id="unique_grid_id4",                       # Unique grid identifier
    #     search_criteria={"strategy": "Cartesian"} # Exhaustive Cartesian grid search
    # )

    # # Train the grid search on the training and validation datasets
    # grid.train(x=predictors, y=response, training_frame=train_h2o, validation_frame=test_h2o)

    # Define the GBM model with early stopping
    gbm = H2OGradientBoostingEstimator(
        seed=1234,
        #nfolds=3,  # Reduce folds to speed up training
        keep_cross_validation_models=False,
        keep_cross_validation_predictions=False,
        keep_cross_validation_fold_assignment=False,
        stopping_rounds=5,  # Enable early stopping
        stopping_metric="rmse",  # Stop when RMSE doesn't improve
        stopping_tolerance=0.001,
        score_each_iteration=True  # Enables scoring logs
    )

    # Define hyperparameter grid
    # hyper_params = {
    #     "ntrees": [100, 300, 500],  # Lowered the max trees for efficiency
    #     "max_depth": [8, 10, 12],  # Reduced range to avoid overfitting
    #     "learn_rate": [0.01, 0.05]  # Removed 0.02 to decrease combinations
    # }
    hyper_params = {
        "ntrees": [197],  # Lowered the max trees for efficiency
        "max_depth": [12],  # Reduced range to avoid overfitting
        "learn_rate": [0.05]  # Removed 0.02 to decrease combinations
    }
    
    # Grid search criteria: Random search to speed up
    search_criteria = {
        "strategy": "RandomDiscrete",
        "max_models": 10,  # Limits total models to prevent long runtime
        "seed": 1234
    }

    # Create and train the grid search
    grid = H2OGridSearch(
        model=gbm,
        hyper_params=hyper_params,
        grid_id="optimized_grid",
        search_criteria=search_criteria
    )

    # Start training
    print("🚀 Starting Grid Search...")
    start_time = time.time()
    grid.train(x=predictors, y=response, training_frame=train_h2o, validation_frame=test_h2o)
    grid_models = grid.get_grid()
    num_grid_models = len(grid_models.models)
    print("Number of grid_models: ",  num_grid_models)

    # Monitor Progress Correctly
    m = 1
    while True:
        try:
            # List models explicitly
            all_models = h2o.ls()  # List all trained models
            print(f"📊 H2O currently has {len(all_models)} models in memory.")

            # If you've reached the max number of models, break
            if m >= num_grid_models:
                print("🔚 Reached the maximum number of models. Exiting loop.")
                break

            m = m + 1

        except Exception as e:
            print(f"⚠️ Error fetching grid status: {e}")

        time.sleep(10)

    # Print total runtime
    end_time = time.time()
    print(f"🎯 Grid Search Completed in {round(end_time - start_time, 2)} seconds!")

    # Print the best model so far
    best_model = grid.get_grid(sort_by="rmse", decreasing=False).models[0]
    print("🏆 Best Model Found:")
    print(best_model)

    # Save logs for debugging
    h2o.download_all_logs(dirname="h2o_logs")
    print("📜 Logs saved to 'h2o_logs/' for debugging.")

    # Print the parameters of the best model
    print("Best Model Parameters:")
    for param, value in best_model.params.items():
        print(f"{param}: {value}")

    best_r2 = best_model.model_performance(test_data=test_h2o).r2()
    print(f"The best R² value is: {best_r2}")

    ########################################


    # Variable Importance
    varimp = best_model.varimp(use_pandas=True)
    print("\nVariable Importance (Top 20):")
    display(varimp.head(20))

    # Function to display plot
    def show_plot(plot_func, title):
        print("Title: ", title)
        plt.figure(figsize=(12, 8))
        try:
            plot_func()
            plt.title(title)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            display(Image(data=buf.getvalue()))
            plt.show()

        except Exception as e:
            print(f"Error generating plot {title}: {str(e)}")
        finally:
            plt.close()

    best_model.varimp_plot()

    # determine SHAP values for model gbm
    #shap_values = gbm.predict(test, type="shap")

    # calculate SHAP values using function predict_contributions
    contributions = best_model.predict_contributions(test_h2o)

    # create dataframe from H2O test
    test_df = test_h2o.as_data_frame(use_pandas=True)

    # add SHAP values to test_df
    shap_values_df = contributions.as_data_frame()

    # give shap_values_df index of df
    shap_values_df.index = test.index.tolist()

        # save shap_values_df to csv
    #shap_values_df.to_csv(path_shap_values, index=True)

    ################################################################

    #test_df = pd.concat([test_df, shap_values_df], axis=1)

    # delete 'encoded_' for all column names in shap_values_df
    for feature in cat_features:
        shap_values_df.rename(columns={f'encoded_{feature}': feature}, inplace=True)

    # save shap_values_df to csv
    shap_values_df.to_csv(path_shap_values, index=True)

    # Convert the SHAP values into a numpy array for the plotting library
    shap_values_np = shap_values_df.values

    # Create the SHAP summary plot
    # Assuming that the last column is the bias term, drop it for the plot
    #shap.summary_plot(shap_values_np[:, :-1], test_df)
    #plt.show()

    # convert the H2O Frame to use with shap's visualization functions
    #shap_values = contributions.as_data_frame().values

    # SHAP Summary Plot
    print("\nSHAP Summary Plot:")
    show_plot(lambda: gbm.shap_summary_plot(test_h2o), "SHAP Summary Plot")

    # Analyze the impact of categorical variables
    encoded_cat_features = [f'encoded_{feature}' for feature in cat_features]
    cat_importance = varimp[varimp['variable'].isin(encoded_cat_features)]['relative_importance'].sum()
    print("\nImpact of categorical variables:")
    print(f"Total importance: {cat_importance}")

    # Get top 5 encoded categorical features by importance
    top_5_encoded = varimp[varimp['variable'].isin(encoded_cat_features)].head(1)['variable'].tolist()

    # PDP Plots for top 5 encoded categorical features
    for feature in top_5_encoded:
        print(f"\nPartial Dependence Plot for {feature}:")
        try:
            show_plot(lambda: gbm.partial_plot(test, cols=[feature], server=False, plot=True, nbins=20), 
                      f"Partial Dependence Plot for {feature}")
        except Exception as e:
            print(f"Error generating Partial Dependence Plot for {feature}: {str(e)}")

    # Function to create ranking dataframe
    def create_ranking_df(encoded_feature, encoding_dict, df):
        original_feature = encoded_feature.replace('encoded_', '')
        rankings = encoding_dict[original_feature]
        
        ranking_df = pd.DataFrame(list(rankings.items()), columns=[original_feature, encoded_feature])
        ranking_df['Median_Total_Costs'] = ranking_df[original_feature].map(df.groupby(original_feature)[target].median())
        
        # Handle any NaN values
        ranking_df['Median_Total_Costs'] = ranking_df['Median_Total_Costs'].fillna(df[target].median())
        
        ranking_df = ranking_df.sort_values(encoded_feature)
        return ranking_df

except Exception as e:
    print(f"An error occurred during the analysis: {str(e)}")

finally:
    # Shutdown H2O cluster
    #h2o.shutdown()
    pass

print("\nAnalysis complete.")

# Function to create ranking dataframe with feature name and count
def create_ranking_df_with_name_and_count(feature, encoding_dict, df):

    original_feature = feature.replace('encoded_', '')
    rankings = encoding_dict[original_feature]

    ranking_df = pd.DataFrame(list(rankings.items()), columns=['Original_Category', 'Rank'])
    
    # Calculate median total costs and count for each category
    grouped_data = df.groupby(original_feature)[target].agg(['median', 'count']).reset_index()
    grouped_data.columns = [original_feature, 'Median_Total_Costs', 'Count']

    ranking_df = ranking_df.merge(grouped_data, left_on='Original_Category', right_on=original_feature, how='left')

    # Handle any NaN values
    ranking_df['Median_Total_Costs'] = ranking_df['Median_Total_Costs'].fillna(df[target].median())
    ranking_df['Count'] = ranking_df['Count'].fillna(0)
    
    ranking_df = ranking_df.sort_values('Rank')
    ranking_df['Feature'] = original_feature
    
    # Reorder columns
    ranking_df = ranking_df[['Feature', 'Rank', 'Median_Total_Costs', 'Original_Category', 'Count']]
    
    return ranking_df

# Create a list to store all ranking dataframes
all_rankings = []

# Create ranking dataframes for each categorical feature
for feature in cat_features:
    encoded_feature = f'encoded_{feature}'
    if encoded_feature in df.columns:
        ranking_df = create_ranking_df_with_name_and_count(encoded_feature, encoding_dict, df)
        all_rankings.append(ranking_df)

# Combine all ranking dataframes
combined_rankings = pd.concat(all_rankings, ignore_index=True)

# Export to CSV
#csv_filename = 'ranked_categorical_features_with_count.csv'
combined_rankings.to_csv(path_combined_ranking_data, index=True)

print(f"Ranked categorical features with count have been exported to {path_combined_ranking}")

# Display the first few rows of the combined rankings
display(combined_rankings.head())

# Print some summary statistics
print("\nSummary Statistics:")
print(f"Total number of features: {combined_rankings['Feature'].nunique()}")
print(f"Total number of categories: {len(combined_rankings)}")
print(f"Average number of categories per feature: {len(combined_rankings) / combined_rankings['Feature'].nunique():.2f}")
print(f"Total rows in dataset: {combined_rankings['Count'].sum()}")
print(f"Average count per category: {combined_rankings['Count'].mean():.2f}")
print(f"Median count per category: {combined_rankings['Count'].median():.2f}")

h2o.shutdown()