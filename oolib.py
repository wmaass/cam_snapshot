import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import joblib  # for saving the model

import functools
import inspect

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from tabulate import tabulate

import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

model_type = 'regression'  # Change to 'regression' for regression

import functools
import time

import logging



# XGboost model for finding performance values before applying CAM
class XGBoostRegressorModel:
    def __init__(self, colsample_bytree=0.8, learning_rate=0.05, max_depth=7, alpha=1, n_estimators=200, subsample=0.8):
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.model = xgb.XGBRegressor(objective='reg:squarederror',
                                       colsample_bytree=self.colsample_bytree,
                                       learning_rate=self.learning_rate,
                                       max_depth=self.max_depth,
                                       alpha=self.alpha,
                                       n_estimators=self.n_estimators)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        metrics = {
            'R2 Score': r2_score(y_test, y_pred),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

def model_train_evaluate(df, target_feature, model_path):
    # Extract features and target variable from the dataframe
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the model class
    xgb_model = XGBRegressor()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'colsample_bytree': [0.3, 0.7, 1],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'alpha': [0, 1, 10],
        'n_estimators': [50, 100, 200]
    }

    # Use GridSearchCV for hyperparameter optimization
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                            scoring='r2', cv=3, verbose=1, n_jobs=-1)

    # Train the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model using joblib
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-Squared: {r2}")

    # Optionally save evaluation metrics and best params to CSV
    evaluation_metrics = {
        'Mean Squared Error': [mse],
        'Mean Absolute Error': [mae],
        'R-Squared': [r2],
        'Best Params': [grid_search.best_params_]
    }

    metrics_df = pd.DataFrame(evaluation_metrics)
    metrics_df.to_csv('evaluation_metrics_with_best_params.csv', index=False)

# -------------

def trace_function(func):
    """A decorator that logs the name of the function being executed and measures its execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Entering: {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Exiting: {func.__name__}")
        logging.info(f"Execution time of {func.__name__}: {end_time - start_time:.6f} seconds")
        return result
    return wrapper


class ConceptContributionAnalyzer:
    def __init__(self, model_type, data, outcome_feature, input_features, concept_names, c, mi, i_id, path, verbose=0, 
                 colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10):
        self.model_type = model_type
        self.data = data
        self.outcome_feature = outcome_feature
        self.input_features = input_features
        self.input_features_names = [j for i in input_features for j in i]
        self.concept_names = concept_names
        self.verbose = verbose
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        #self.model = None
        self.r2 = None
        self.shap_values = None
        self.means = None
        self.c = c
        self.mi = mi
        self.i_id = i_id
        self.res_path = path
        self.positive_values = pd.Series()
        self.positive_cumulative_percentage = 0
        self.negative_values = pd.Series()
        self.negative_cumulative_percentage = 0
        #
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.model = xgb.XGBRegressor(objective='reg:squarederror',
                                       colsample_bytree=self.colsample_bytree,
                                       learning_rate=self.learning_rate,
                                       max_depth=self.max_depth,
                                       alpha=self.alpha,
                                       n_estimators=self.n_estimators)

    def preprocess_data(self):
        #input = [j for i in self.input_features for j in i]
        self.X = self.data[self.input_features_names]
        self.y = self.data[self.outcome_feature]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)

    # def train_model(self):
    #     if self.model_type == 'classification':
    #         self.model = xgboost.XGBClassifier()
    #     else:
    #         from sklearn.model_selection import GridSearchCV
    #         import multiprocessing
    #         model = GradientBoostingRegressor()
    #         reg = GridSearchCV(model, {'max_depth': [2], 'n_estimators': [50]}, verbose=0,
    #                            n_jobs=multiprocessing.cpu_count() // 2)
    #         reg_fit = reg.fit(self.X_train, self.y_train)
    #         self.model = GradientBoostingRegressor(**reg.best_params_)
    #         self.model.fit(self.X_train, self.y_train)

    # function that replaces one feature with white noise
    def replace_feature_with_noise(self, data, feature):
        data_copy = data.copy()
        data_copy[feature] = np.random.normal(data_copy[feature].mean(), data_copy[feature].std(), data_copy[feature].shape)
        return data_copy


    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        self.r2 = r2_score(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        metrics = {
            'R2 Score': self.r2,
            'Mean Absolute Error': self.mae,
            'Mean Squared Error': self.mse,
            'Root Mean Squared Error': self.rmse
        }

        if not os.path.isfile(f"{self.res_path}performance.csv"):
            with open(f"{self.res_path}performance.csv", "w") as f:
                f.write("c,mi,i_id,r2,mae,rmse,input_features\n")

        # append performance values to self.res_path folder in a csv file; write float values with 3 decimal places
        with open(f"{self.res_path}performance.csv", "a") as f:
            f.write(f"{self.c},{self.mi},{self.i_id},{self.r2:.3f},{self.mae:.3f},{self.mse:.3f},{self.rmse:.3f},{self.input_features_names}\n")

        return metrics
    
    def train_model_2(self):
        gpu_params = {
            'predictor': 'gpu_predictor',
            'tree_method': 'hist',
            'device': 'gpu',
            'objective': 'reg:squarederror' if self.model_type != 'classification' else 'binary:logistic'
        }

        # print feature names of X_train
        print("X_train: ", self.X_train.columns)

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)

        # Define the parameter grid
        param_grid = {
            'max_depth': [2], #, 4, 6],
            'n_estimators': [50], #, 100, 150],
            'learning_rate': [0.01] #, 0.1, 0.2]
        }

        min_rmse = float("Inf")
        best_params = None

        # Manual grid search
        for max_depth in param_grid['max_depth']:
            for n_estimators in param_grid['n_estimators']:
                for lr in param_grid['learning_rate']:
                    params = gpu_params.copy()
                    params.update({
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'learning_rate': lr
                    })
                    
                    # xgboost cross-validation with early stopping
                    cv_results = xgb.cv(
                        params,
                        dtrain,
                        num_boost_round=10,
                        nfold=5,
                        metrics='rmse',
                        early_stopping_rounds=10
                    )

                    mean_rmse = cv_results['test-rmse-mean'].min()
                    if mean_rmse < min_rmse:
                        min_rmse = mean_rmse
                        best_params = params

        if best_params:
            self.model = xgb.train(best_params, dtrain, num_boost_round=10)
        else:
            print("Warning: Failed to find suitable model parameters. Using default parameters.")
            self.model = xgb.train(gpu_params, dtrain, num_boost_round=10)


    @trace_function
    # def predict(self, X_test):
    #     dtest = xgb.DMatrix(X_test)
    #     if self.model is not None:
    #         y_pred = self.model.predict(dtest)
    #         return y_pred
    #     else:
    #         raise Exception("Model is not trained yet")
        
    # def evaluate_model(self):
    #         if self.model_type == 'classification':

    #             # Convert your test data into DMatrix
    #             dtest = xgb.DMatrix(self.X_test)

    #             # Make predictions
    #             y_pred = self.model.predict(dtest)

    #             accuracy = accuracy_score(self.y_test, y_pred)
    #             precision = precision_score(self.y_test, y_pred)
    #             recall = recall_score(self.y_test, y_pred)
    #             f1 = f1_score(self.y_test, y_pred)

    #             print("Classification Metrics:")
    #             print(f"Accuracy: {accuracy:.2f}")
    #             print(f"Precision: {precision:.2f}")
    #             print(f"Recall: {recall:.2f}")
    #             print(f"F1-Score: {f1:.2f}")
    #         else:  # Regression
                
    #             # Convert your test data into DMatrix
    #             dtest = xgb.DMatrix(self.X_test)

    #             # Make predictions
    #             y_pred = self.model.predict(dtest)

    #             r2 = metrics.r2_score(self.y_test, y_pred)
    #             mae = metrics.mean_absolute_error(self.y_test, y_pred)
    #             rmse = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))

    #             # if performance.csv does not exist, create it and write header

    #             if not os.path.isfile(f"{self.res_path}performance.csv"):
    #                 with open(f"{self.res_path}performance.csv", "w") as f:
    #                     f.write("c,mi,i_id,r2,mae,rmse,input_features\n")

    #             # append performance values to self.res_path folder in a csv file; write float values with 3 decimal places
    #             with open(f"{self.res_path}performance.csv", "a") as f:
    #                 f.write(f"{self.c},{self.mi},{self.i_id},{r2:.3f},{mae:.3f},{rmse:.3f},{self.input_features_names}\n")

    #             #print("Regression Metrics:")
    #             #print(f"R-squared (R2): {r2:.2f}")
    #             #print(f"Mean Absolute Error (MAE): {mae:.2f}")
    #             #print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


    def pareto_plot(self):
        # Creating a figure with two subplots arranged horizontally
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Plotting Positive Values
        positive_ax = ax1
        positive_ax.bar(self.positive_values.index, self.positive_values, color='green', label='Value')
        positive_ax.set_xlabel('Categories')
        positive_ax.set_ylabel('Value', color='tab:green')
        positive_ax.tick_params(axis='y', labelcolor='tab:green')
        positive_ax.set_xticklabels(self.positive_values.index, rotation=90)
        positive_ax.set_title('Pareto Plot - Positive Values')

        positive_ax2 = positive_ax.twinx()
        positive_ax2.plot(self.positive_values.index, self.positive_cumulative_percentage, color='red', marker='o', linestyle='-', linewidth=2, markersize=5)
        positive_ax2.set_ylabel('Cumulative Percentage', color='tab:red')
        positive_ax2.tick_params(axis='y', labelcolor='tab:red')
        positive_ax2.axhline(80, color='grey', linestyle='dashed')
        positive_ax2.text(0, 85, '80% Threshold', color='grey')
        positive_ax2.set_ylim(0, 100)  # Set y-axis to range from 0% to 100%

        # Plotting Negative Values
        negative_ax = ax2
        negative_ax.bar(self.negative_values.index, self.negative_values, color='blue', label='Value')
        negative_ax.set_xlabel('Categories')
        negative_ax.set_ylabel('Value', color='tab:blue')
        negative_ax.tick_params(axis='y', labelcolor='tab:blue')
        negative_ax.set_xticklabels(self.negative_values.index, rotation=90)
        negative_ax.set_title('Pareto Plot - Negative Values')

        negative_ax2 = negative_ax.twinx()
        negative_ax2.plot(self.negative_values.index, self.negative_cumulative_percentage, color='red', marker='o', linestyle='-', linewidth=2, markersize=5)
        negative_ax2.set_ylabel('Cumulative Percentage', color='tab:red')
        negative_ax2.tick_params(axis='y', labelcolor='tab:red')
        negative_ax2.axhline(80, color='grey', linestyle='dashed')
        negative_ax2.text(0, 85, '80% Threshold', color='grey')
        negative_ax2.set_ylim(0, 100)  # Set y-axis to range from 0% to 100%

        fig.tight_layout()

        # Save plot
        plt.savefig(f"{self.res_path}pareto_plot_{self.c}_{self.mi}_{self.i_id}.png")

        plt.close()

    @trace_function
    def explain_model(self):
        if self.model is not None:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.X_test, check_additivity=False)

            stats = pd.DataFrame(self.shap_values).describe()

            self.means = stats.loc[["mean"]]
            #self.means = self.means.sort_values(by='mean', ascending=False, axis=1)

            c = self.X_test.iloc[:, self.means.columns].columns
            self.means = self.means.set_axis(c, axis=1, copy=False)

            # create dataframe with self.means as values
            means_df = pd.DataFrame(self.means)

            means_df.columns = self.input_features_names

            # Filter out columns with value 0
            non_zero_columns = means_df.loc[:, (means_df != 0).any(axis=0)]

            self.positive_values = non_zero_columns.iloc[0][non_zero_columns.iloc[0] > 0].sort_values(ascending=False)
            self.negative_values = non_zero_columns.iloc[0][non_zero_columns.iloc[0] < 0].sort_values()

            # Calculate cumulative percentage for both positive and negative values
            self.positive_cumulative_percentage = self.positive_values.cumsum() / self.positive_values.sum() * 100
            self.negative_cumulative_percentage = self.negative_values.cumsum() / self.negative_values.sum() * 100

            self.pareto_plot()
            
            # Plot SHAP summary plot
            #shap.summary_plot(self.shap_values, self.X_test, show=False)
            #plt.title("SHAP Summary Plot")
            #plt.show()

            # save plot to self.res_path folder
            #plt.savefig(f"{self.res_path}shap_summary_plot_{self.c}_{self.mi}_{self.i_id}.png")

            #plt.close()
            

            # Individual SHAP value plot (e.g., for the first data point)
            #shap.initjs()
            #shap.force_plot(explainer.expected_value, self.shap_values[0], self.X_test.iloc[0, :], matplotlib=True)
        else:
            print("Model is not trained.")

    @trace_function
    def calculate_concept_contributions(self):
        g_c = []
        a_c = []
        for c in self.input_features:
            # Calculate the mean of SHAP values for the columns in c
            shap_values_subset = [self.shap_values[:, self.X_test.columns.get_loc(col)] for col in c]

            # Convert the list of lists to a DataFrame. Each inner list becomes a column.
            df = pd.DataFrame(shap_values_subset).transpose()  
            
            # Calculate the mean of each column
            column_means = df.mean()

            # Filter the means to only include those greater than 0
            positive_means = column_means[column_means != 0]

            # Calculate the sum of these positive means
            sum_positive_means = positive_means.sum()

            # Count the number of columns with mean greater than 0
            num_columns_gt_zero = positive_means.count()

            # Calculate the average of the positive means
            average_of_positive_means = sum_positive_means / num_columns_gt_zero if num_columns_gt_zero else 0

            # save to csv file
            #shap_values_df = pd.DataFrame(shap_values_subset)
#            shap_values_df.to_csv(f"{self.res_path}_shap_values_subset_{self.c}_{self.mi}_{self.i_id}.csv", index=False)

            #s = np.mean(np.array(shap_values_subset))
            #g_c.append(s)
            g_c.append(average_of_positive_means)
            a_c.append(shap_values_subset)

        e = [self.outcome_feature] + g_c

        # merge lists self.concept_names[0] and .concept_names[1:len(self.input_features) + 1] into one list
        cols = [self.concept_names[0]] + self.concept_names[1]

        gc_df = pd.DataFrame([e])

        gc_df.columns = cols

        return gc_df

    @trace_function
    def analyze_concept_contribution(self):
        self.preprocess_data()
        #self.replace_feature_with_noise(self.X_train, self.input_features[0][0])
        # self.train_model()
        # self.evaluate_model()

        self.train()
        self.evaluate()

        self.explain_model()

        if self.model is not None:
            explainer = shap.TreeExplainer(self.model)

            self.shap_values = explainer.shap_values(self.X_test, check_additivity=False)

            # save self.shap_values to csv file with path self.res_path
            shap_values_df = pd.DataFrame(self.shap_values, index=self.X_test.index)

            # set column names of shap_values_df to column names of self.X_test
            shap_values_df.columns = self.X_test.columns

            #shap_values_df.to_csv(f"{self.res_path}_shap_values_df_{self.c}_{self.mi}_{self.i_id}.csv", index=False)


            # save sh_df to csv file with name sh_df plus c 
            #sh_df.to_csv(path_results + 'sh_df' + c + '.csv')
            shap_values_df.to_csv(self.res_path + 'shap_values_df.csv')

            self.X_test.to_csv(self.res_path + 'X_test.csv')

            # print first element of self.X_test
            print(self.y.iloc[0])

            # select index from sample of self.y_test with value larger than 100000 (HIS17) or 0 (LOAN)
            idx = np.where(self.y_test > 0)[0][0]

            print(idx)
            print(self.X_test.iloc[idx,:])
            print(self.y_test.iloc[idx])

            # Create an Explanation object
            explanation = shap.Explanation(
                values=self.shap_values[idx],              # SHAP values for the selected sample
                base_values=explainer.expected_value,      # Expected value
                data=self.X_test.iloc[idx, :],             # Feature values for the selected sample
                feature_names=self.X_test.columns          # Column names
            )

            # Create a figure with two subplots, giving more width to the waterfall plot (ax1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [3, 1]})  # Set the width ratio to give ax1 more space

            # Plot the SHAP waterfall plot on the left side (ax1)
            plt.sca(ax1)  # Set the current axis to ax1
            shap.waterfall_plot(explanation, show=False)  # Plot the waterfall plot on ax1

            # Adjust font size of the waterfall plot manually
            for text in ax1.get_yticklabels():
                text.set_fontsize(8)
            for text in ax1.get_xticklabels():
                text.set_fontsize(8)
            ax1.title.set_fontsize(8)
            ax1.set_title("SHAP Waterfall Plot", fontsize=8)

            # Create textbox content from explanation data and corresponding feature names
            text = "\n".join([f"{explanation.feature_names[i]}: {explanation.data[i]}" for i in range(len(explanation.data))])

            # Plot text to the right side (ax2)
            ax2.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            ax2.axis('off')  # Turn off the axis for the text box

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(f"{self.res_path}shap_waterfall_plot_{self.c}_{self.mi}_{self.i_id}.png")
            plt.show()


            # save self.shap_values to csv file
            #shap_values_df = pd.DataFrame(self.shap_values)
            #shap_values_df.to_csv(f"{self.res_path}_shap_values.csv", index=False)

            concept_contributions = self.calculate_concept_contributions()

            dtest = xgb.DMatrix(self.X_test)


            #r2 = metrics.r2_score(self.y_test, self.model.predict(dtest))
            return concept_contributions, self.shap_values, self.means, self.X_test, self.r2
        else:
            r2 = None  # Handle the case where the model is not trained
            return None, None, r2

class FeatureSeparator:
    def __init__(self, cd, non_input_concepts, path_results, path_figures, epsilon=0.1, cluster_cut=0.1, concept_max=0):
        self.cd = cd
        self.non_input_concepts = non_input_concepts
        self.path_results = path_results
        self.path_figures = path_figures
        self.epsilon = epsilon
        self.cluster_cut = cluster_cut
        self.concept_max = concept_max

    def minorclass3(self, features, labels):
        # Determine the label with the most occurrences
        unique, counts = np.unique(labels, return_counts=True)
        most_common_label = unique[np.argmax(counts)]

        # Get the indices of attributes not associated with the most common label
        indices = np.where(labels != most_common_label)[0]

        # Return the corresponding attributes
        return [features[i] for i in indices]

    # Function to add a new row with specified values to the given features

    def add_row_with_values(self, df, f, values, c, mi, iter, idx_ai):
    #self.add_row_with_values(attribute_iterations, f, clusters, c, mi, ci, idx_ai)

        # Create a new row with NaN values for all columns
        new_row = pd.Series({col: np.nan for col in df.columns}, name='new_row')
        
        # Update the new row with the provided values for the specified features
        for feature, value in zip(f, values):
            if feature in new_row.index:
                new_row[feature] = value

        new_row['idx'] = iter
        new_row['concept'] = c
        new_row['main_iter'] = mi
        new_row['iter'] = idx_ai

        # Append the new row to the dataframe using pd.concat
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # cast new_row into dataframe
        n = pd.DataFrame(new_row).transpose()

        # if att_iter.csv does not exist, create it and write df to it with header else without header
        if not os.path.isfile(self.path_results + 'att_iter.csv'):
            n.to_csv(self.path_results + 'att_iter.csv', mode='a', header=True)
        else:
            n.to_csv(self.path_results + 'att_iter.csv', mode='a', header=False)

        return df

    @trace_function
    def separate_features(self, attribute_iterations, mi, ci, idx_ai):
        df = pd.read_csv(self.path_results + 'GC_all.csv', delimiter=',', quotechar='"', encoding='latin1')
        df.rename(columns={'Unnamed: 0': 'target'}, inplace=True)

        # replace NaN with 0
        df = df.fillna(0)

        c_labels = []
        df_c = pd.DataFrame()
        idx = 0
        f_labels = pd.DataFrame(columns=df.columns)
        f_labels.loc[len(f_labels)] = [0] * len(df.columns)
        f_labels.loc[len(f_labels)] = [0] * len(df.columns)
        cd_labels = {}
        unique_index = df['target'].unique()
        df = df.iloc[:, 2:-2]
        df = df.fillna(0)
        sep_feats = []
        ts_X = {}

        for c in self.cd:
            X = []
            if (c not in self.non_input_concepts) and (len(self.cd[c]) > 1):
                index_names = df.index.values
                length = len(self.cd[c])

                count_cd = 0
                for f in self.cd[c]:
                    count_cd += 1

                idx = 0

                for f in self.cd[c]:
                    ts_g = []
                    for g in df[f]:
                        if (not np.isnan(g)) and (isinstance(g, float)):
                            ts_g.append(g)
                        else:
                            print("NaN or not float: ", g)
                    X.append(list(ts_g))
                    ts_X[f] = X[idx]
                    idx = idx + 1

                # data contains all vectors of the attributes associated with concept self.cd[c] for predicting all other concepts
                data = np.array(X)
                data_T = data
                data_T = data_T.astype(np.float64)
                column_names = self.cd[c]
                
                wcss = []  # Create an empty list to store the within-cluster sum of squares (WCSS)
                
                max_clusters = data_T.shape[0]
                
                non_zero_cols = np.any(data_T != 0, axis=0)
                filtered_data = data[:, non_zero_cols]

                # Determine the optimal number of clusters
                for i in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10, max_iter=10)
                    kmeans.fit(data_T)
                    wcss.append(kmeans.inertia_)

                knee_locator = KneeLocator(range(1, max_clusters + 1), wcss, curve='convex', direction='decreasing')
                optimal_clusters = knee_locator.elbow

                if optimal_clusters is None:
                    optimal_clusters = 1

                # Clustering by K-means++ (and PCA - what is it good for?)
                    
                # optimal_clusters = 2

                kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
                kmeans.fit_predict(data_T)

                clusters = kmeans.labels_
                
                K = range(1, count_cd+1)

                cd_labels[c] = clusters

                # select columns from dataframe 'attribute_iterations' with column names in self.cd[c]
                # and set all values to values in clusters

                f = self.cd[c]
                # cast f into list
                f = list(f)

                attribute_iterations = self.add_row_with_values(attribute_iterations, f, clusters, c, mi, ci, idx_ai)

            if len(self.cd[c]) == 1:
                cd_labels[c] = [0]
        
        count = [0] * df.shape[1]
        feat_list = []
        concept_list = []

        for c, flist in self.cd.items():
            if c not in self.non_input_concepts:
                for f in flist:
                    concept_list.append(c)
                    feat_list.append(f)

        if len(df.columns) != len(feat_list):
            s1 = set(df.columns)
            s2 = set(feat_list)
            s = s1 - s2
            print("Not in feat_list: ", s)
            s = s2 - s1
            print("Not in df: ", s)

            print(f"Not in df and feat_list: {attributes_not_in_both(df.columns, feat_list)}")

        df.columns = feat_list
        df.loc[-1] = concept_list
        df.index = df.index + 1
        df.sort_index(inplace=True)
        df.to_csv(self.path_results + 'features_concepts.csv', mode='a')

        for c in cd_labels:
            if c not in self.non_input_concepts:
                pos = 0
                features = list(self.cd[c])
                labels = cd_labels[c]

                sep_c_f = self.minorclass3(features, labels)

                for i in range(0, len(df.columns)):
                    if df.columns[i] in sep_c_f:
                        count[i] = -1
                        sep_feats.append(df.columns[i])
                        print("Concept: ", c, " Feature: ", df.columns[i], " Label: ", labels[pos])
                        print(sep_feats)
                        print("############################################")

        count = pd.DataFrame(count)
        count = count.transpose()
        count.columns = df.columns
        count.to_csv(self.path_results + 'count.csv', mode='a')

        return (sep_feats, ts_X)

# Usage:
# Create an instance of FeatureSeparator and call separate_features() to perform the separation


class DataHandler:
    def __init__(self, base_path):
        self.base_path = base_path

    def read_csv(self, file_name, index_col=0):
        return pd.read_csv(f'{self.base_path}/{file_name}', index_col=index_col)

    def process_rdata(self, rdata):
        rdata = rdata.iloc[::-1].reset_index(drop=True)
        rdata.index += 1
        rdata.loc[0] = 0
        return rdata.sort_index()

    def process_concepts(self, concepts):
        concepts['concept'] = concepts.index.copy()
        concepts = concepts.reset_index(drop=True)
        concepts['iteration'] = 0
        i = 1
        for index, row in concepts.iterrows():
            if row['concept'] == '*iteration*':
                i += 1
            concepts.loc[index, 'iteration'] = i

        return concepts[concepts['concept'] != '*iteration*']


class GraphPlotter:
    def __init__(self, label_offset=0.1, threshold=0.15, edge_offset=0.05):
        self.label_offset = label_offset
        self.threshold = threshold
        self.edge_offset = edge_offset

    def offset_label_pos(self, pos):
        return {node: (x + self.label_offset, y + self.label_offset) for node, (x, y) in pos.items()}

    def offset_edge_pos(self, pos, edge, edge_offset):
        # Function to create an offset position for a specific edge
        (u, v) = edge
        (x_u, y_u), (x_v, y_v) = pos[u], pos[v]
        return (x_u + edge_offset, y_u + edge_offset), (x_v + edge_offset, y_v + edge_offset)

    def plot_graph(self, path, dfi, rdatai, iteration):
        plt.figure(figsize=(8, 8))
        G = nx.DiGraph()

        for index, row in dfi.iterrows():
            if abs(row['gcbm']) >= self.threshold:
                G.add_edge(row['from'], row['to'], weight=row['gcbm'])

        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue')
        
        # Draw all edges
        for u, v, data in G.edges(data=True):
            color = 'red' if data['weight'] < 0 else 'black'
            if color == 'red':
                # Offset position for red edges
                pos_edge = self.offset_edge_pos(pos, (u, v), self.edge_offset)
                nx.draw_networkx_edges(G, {u: pos_edge[0], v: pos_edge[1]}, edgelist=[(u, v)], width=1, arrows=True, edge_color=color)
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, arrows=True, edge_color=color)

        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        offset_label_pos = self.offset_label_pos(pos)
        for node in G.nodes():
            for col in rdatai.index:
                if col == node:
                    label = f"R2: {round(rdatai[col], 2)}"
                    nx.draw_networkx_labels(G, offset_label_pos, labels={node: label}, font_size=8, font_color='b')

        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

        # Calculate margins for the plot
        x_values, y_values = zip(*pos.values())
        x_margin = (max(x_values) - min(x_values)) * 0.30  # 20% margin
        y_margin = (max(y_values) - min(y_values)) * 0.30  # 20% margin

        plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        plt.axis('off')
        plt.savefig(f"{path}/GCBM{iteration}.png", dpi=300, bbox_inches='tight')
        plt.show()



class FileWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_iteration(self, path, iteration, content):
        full_file_path = os.path.join(path, self.file_name)
        with open(full_file_path, "a") as file:
            file.write(f"Iteration: {iteration}\n{content}\n")
