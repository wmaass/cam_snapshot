# Licensed to Wolfgang Maass, Saarland University, under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information 
# regarding ownership and licensing. Wolfgang Maass, Saarland University, licenses this file to 
# you under the Wolfgang Maass License (the "License"); you may not use this 
# file except in compliance with the License. You may obtain a copy of the License at:
#
#     http://www.uni-saarland.de/legal/license
#
# Unless required by applicable law or agreed to in writing, software distributed 
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
# CONDITIONS OF ANY KIND, either express or implied. See the License for the 
# specific language governing permissions and limitations under the License.
#
# This file is part of the research conducted by Prof. Wolfgang Maass, 
# at Saarland University, in the domain of artificial intelligence. 
# The code implements AI models, which is relevant for research 
# in Explainable AI (XAI).
#
# © 2024, Wolfgang Maass, Saarland University. All rights reserved.

import os
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    adjusted_rand_score, 
    adjusted_mutual_info_score, 
    normalized_mutual_info_score, 
    homogeneity_score, 
    completeness_score, 
    v_measure_score,
    silhouette_score,
    confusion_matrix
)
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from scipy.stats import ks_2samp
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind

import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import openai

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table, State, callback_context
import dash_bootstrap_components as dbc
import webbrowser
from threading import Timer

from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from collections import Counter

import json

import plotly.graph_objects as go

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

from flask import request

import re

#app_name = "HIS17-two-classes-d" 
#target = "Gesamtkosten"

# Initialize H2O
h2o.init()

app_name = "HIS17-two-classes" 

target = "Total Costs"

cluster_num = 4
#cluster_type = 'kmeans'
cluster_type = 'hmeans'

#x_cut = 0.02
x_cut = 500
y_cut = 5 

# Data Access - Statewide Planning and Research Cooperative System (SPARCS)
# https://health.data.ny.gov/dataset/Hospital-Inpatient-Discharges-SPARCS-De-Identified/22g3-z7e7/about_data

# Attempt to read the CSV file from the specified path
#file_path = './results/HIS17-random/sh_dfdecision.csv'
#file_path = './results/'+app_name+'/shap_mlsmall.csv'

result_path = './results/'

file_path = './results/'+app_name+'/ranked/shap_values_df.csv'

#file_path_db = './data/hdma_1M.csv'
file_path_db = './results/'+app_name+'/ranked/test.csv'

inputdata_path_db = './results/' + app_name + '/ranked/rank_encoded.csv'

gbm_model_path = './results/'+app_name+'/ranked/model/'+'GBM_model_python_1739459721817_1'

original_path_db = './results/'+app_name+'/ranked/df_original.csv'

# Create output directory if it doesn't exists
figure_path = './figures/'+app_name
os.makedirs(figure_path, exist_ok=True)

# Create output directory if it doesn't exist
dist_path = './figures/'+ app_name + '/distributions'
os.makedirs(dist_path, exist_ok=True)

################################


def prepare_firework_data(df, feature, prominence):
    # shap_val = np.sign(df[feature]) * np.log2(np.abs(df[feature]) + 1)  # Log2 scale preserving sign
    # m = df[feature].mean()
    shap_val = df[feature] # - m  # Log2 scale preserving sign
    #log2_y = np.log2(np.abs(df[feature] / df[feature].mean()) + 1)

    sh_mean = shap_val.mean()

    #shap_val = np.sign(df[feature]) * (df[feature] - df[feature].mean())  # Log2 scale preserving sign
    log2_y = np.log2(np.abs(shap_val / sh_mean))
    shap_val += np.random.normal(0, 1e-6, size=shap_val.shape)  # Adding tiny noise
    
    firework_df = pd.DataFrame({'shap_val': shap_val, 'Log2_Y': log2_y, 'sh_mean':sh_mean, 'df_index': df.index.values})

    density = gaussian_kde(shap_val)
    xs = np.linspace(min(shap_val), max(shap_val), 500)
    density_values = density(xs)
    peak_classification, peaks_x = classify_peaks(density_values, xs, prominence)
    return firework_df, peak_classification, peaks_x

# Function to classify distribution peaks using find_peaks with adjustable prominence
def classify_peaks(density_values, xs, prominence):
    peaks, _ = find_peaks(density_values, prominence=prominence)
    peaks_x = xs[peaks]
    if len(peaks_x) > 1 and any(peaks_x < 0) and any(peaks_x > 0):
        return "Two Peaks", peaks_x
    return "Single Peak", peaks_x


def calculate_x_div(firework_df, x_t, y_t):
    """
    Calculate the smallest positive value of 'shap_val' with 'log2_y' > y_t (x_div_pos) and
    the largest negative value of 'shap_val' with 'log2_y' > y_t (x_div_neg).
    
    Args:
    - firework_df (pd.DataFrame): DataFrame containing 'shap_val' and 'log2_y' columns.
    - x_t (float): Threshold value for scaling x_div.
    - y_t (float): Threshold value for filtering log2_y.

    Returns:
    - x_div_pos (float): Normalized smallest positive value of 'shap_val' where 'log2_y' > y_t.
    - x_div_neg (float): Normalized largest negative value of 'shap_val' where 'log2_y' > y_t.
    """

    y_div_pos = 0
    y_div_neg = 0
    
    # determine Log2_y for smallest shap_val > x_t
    filtered_df = firework_df[firework_df['shap_val'] > x_t]

    # if filtered_df is not None
    if not filtered_df.empty:
        # sort filtered_df
        filtered_df.sort_values('shap_val', ascending=True, inplace=True)
        y_val_pos = filtered_df['Log2_Y'].iloc[0]
        y_div_pos = y_val_pos / y_t
    else:
        print("filtered_df pos is empty")

    # determine Log2_y for smallest shap_val > x_t
    filtered_df = firework_df[firework_df['shap_val'] < -x_t]

    # y_val = last element of filtered_df
    if not filtered_df.empty:
        # sort filtered_df
        filtered_df.sort_values('shap_val', ascending=False, inplace=True)
        y_val_neg = filtered_df['Log2_Y'].iloc[0]
        y_div_neg = y_val_neg / y_t
    else:
        print("filtered_df neg is empty")
    
    return y_div_pos, y_div_neg

def read_and_collect_features(file_path, thres):
    """
    Reads the 'classification.json' file into a DataFrame and collects feature names 
    based on specific conditions.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    - thres (float): The threshold value for filtering based on 'DivPos' and 'DivNeg'.
    
    Returns:
    - List[str]: A list of feature names that meet the specified conditions.
    """
    # Read the JSON file into a DataFrame
    c_df = pd.read_json(file_path)

    # Initialize a list to collect feature names
    collected_features = []

    # Iterate through the DataFrame rows
    for index, row in c_df.iterrows():
        # Check if the conditions are met
        if (abs(row['DivPos']) > thres or abs(row['DivNeg']) > thres) or (row['DivPos'] == 0 and row['DivNeg'] == 0):
            # Collect the feature name
            collected_features.append(row['Feature'])
    
    return collected_features

# Function to compute peak classifications for all features
def compute_all_peak_classifications(df, prominence, y_t):
    classifications = []
    for feature in df.columns:
        firework_df, classification, _ = prepare_firework_data(df, feature, prominence)

        div_x_pos, div_x_neg = calculate_x_div(firework_df, x_cut, y_t)

        classifications.append({'Feature': feature, 'Classification': classification, 'DivPos': div_x_pos, 'DivNeg': div_x_neg})
    
    # add series divergence_index to classifications as new column
    #classifications = classifications.append(divergence_index.to_frame('Divergence Index'))

    return classifications

def combined_firework_plot(df, prominence, selected_features, x_range, y_cut):
    # Initialize an empty figure for combining multiple firework plots
    combined_fig = go.Figure()
    colors = px.colors.qualitative.Plotly  # Use Plotly's color palette for differentiation

    for i, feature in enumerate(selected_features):
        # Prepare data for the firework plot using the custom preparation function
        firework_df, _, _ = prepare_firework_data(df, feature, prominence)
        
        # Adding check to ensure firework_df contains the required columns
        if 'shap_val' not in firework_df.columns or 'Log2_Y' not in firework_df.columns:
            print(f"Error: Missing expected columns in data for feature '{feature}'. Skipping.")
            continue

        # Create a customdata array with both sh_mean and df_index
        firework_df['customdata'] = list(zip(firework_df['df_index'], firework_df['sh_mean']))

        # Add scatter trace for each feature's firework plot to the combined figure
        combined_fig.add_trace(
            go.Scatter(
                x=firework_df['shap_val'],
                y=firework_df['Log2_Y'],
                customdata=firework_df['customdata'],  # Pass sh_mean and df_index as customdata
                mode='markers',
                marker=dict(color=colors[i % len(colors)], size=3, opacity=0.7),
                name=f'{feature}',  # Label for the feature
                hovertemplate=(
                    f"<b>{feature}</b><br>" +
                    "x: %{x:.2f}, " +  # Rounds x to 2 decimal places
                    "y: %{y:.2f}<br>" +  # Rounds y to 2 decimal places
                    "sh_mean: %{customdata[1]:.2f}<br>" +  # Access sh_mean from customdata
                    "index: %{customdata[0]}<br>" +  # Access df_index from customdata
                    "<extra></extra>"  # Removes trace name from hover info
                )
            )
        )
    # add a dashed horizontal line for y = y_cut
    combined_fig.add_hline(y=y_cut, line=dict(dash='dash', color='red')) 
    # add a dashed vertical line for x = x_cut
    combined_fig.add_vline(x=x_cut, line=dict(dash='dash', color='red'))
    # add a dashed vertical line for x = -x_cut
    combined_fig.add_vline(x=-x_cut, line=dict(dash='dash', color='red'))
    
    # Updating layout of the combined figure with a legend
    combined_fig.update_layout(
        xaxis_title='Feature Attribution Values sf - mean(sf)',
        yaxis_title='Log2(sf/mean(sf))',
        template='plotly_white',
        xaxis=dict(range=x_range),  # Set x-axis range based on provided x_range
        showlegend=True,  # Enable legend
        legend=dict(
            title="Features",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemsizing="constant",  # Ensures consistent legend item size
            font=dict(size=12),  # Adjust font size if needed
            itemwidth=40  # Increase marker size in legend
        )
    )

    return combined_fig

def collect_indexes_around_values(df, feature, most_frequent_values, peak_epsilon):

    # Calculate the average distance between consecutive elements in df[feature]
    sorted_feature_values = df[feature].sort_values().values
    if len(sorted_feature_values) > 1:
        average_distance = np.mean(np.diff(sorted_feature_values))
        epsilon = peak_epsilon * average_distance
    else:
        epsilon = 0.1  # fallback in case of a very small dataset

    # Initialize a dictionary to store indexes for each most frequent value
    indexes_dict = {}
    
    # Iterate over each value in most_frequent_values
    for value in most_frequent_values:
        # Find indexes where the feature value is within the epsilon range of the current value
        indexes = df[(df[feature] >= value - epsilon) & (df[feature] <= value + epsilon)].index.tolist()
        
        # Store the indexes in the dictionary with the current value as the key
        indexes_dict[value] = indexes
    
    return indexes_dict

from collections import Counter

def analyze_frequent_values(fn_db, selected_facility, selected_feature, indexes_dict):

    # Filter df_db to only include rows where 'Facility Name' matches the selected facility
    if selected_facility is not None and selected_facility != 'All':
        df_filtered = fn_db[fn_db['Facility Name'] == selected_facility]
    else:
        df_filtered = fn_db.copy()
    
    # Initialize an empty list to store the final results for the table
    table_data = []
    
    # Iterate over the dictionary of indexes
    for most_frequent_value, indexes in indexes_dict.items():

        # Find matching indexes between df_filtered and the list of indexes
        matching_indexes = df_filtered.index.intersection(indexes)

        # Filter df_filtered based on the matching indexes
        df_matched = df_filtered.loc[matching_indexes]  

        values = df_matched[selected_feature]

        value_counts = Counter(values)

        # Get the 3 most common values
        most_common_values = value_counts.most_common(5)
        
        # Add the most common values to the table data
        for value, count in most_common_values:
            table_data.append({
                'Most Frequent Value': most_frequent_value,
                'Feature Value': value,
                'Frequency': count
            })
        # Add an empty row as a separator after each iteration
        table_data.append({
            'Most Frequent Value': '',  # Empty value to serve as a separator
            'Feature Value': '',
            'Frequency': ''
        })
    
    # Convert the table data into a DataFrame for better visualization
    result_df = pd.DataFrame(table_data)
    
    return result_df

# Function to wrap text based on a specified width in cm
def wrap_text(text, max_width_cm, font_size=12):
    # Convert cm to pixels (1 cm ≈ 37.8 pixels at 96 DPI)
    max_width_px = max_width_cm * 37.8
    words = text.split()
    wrapped_lines = []
    current_line = ""

    for word in words:
        # Check if adding this word would exceed the maximum width
        if len(current_line) + len(word) + 1 > max_width_px // (font_size / 2):  # Approximate character width
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line += " " + word if current_line else word

    if current_line:
        wrapped_lines.append(current_line)

    return "<br>".join(wrapped_lines)

# Function to parse entries into a dictionary
def parse_hovertext(entries, threshold_percent=5):
    parsed_dict = {}

    # Extract values first to compute the sum of absolute values
    extracted_values = []
    for entry in entries:
        if entry:
            match = re.match(r"(.+): \$(-?\d{1,3}(?:,\d{3})*\.\d{2})", entry)
            if match:
                key, value = match.groups()
                numeric_value = float(value.replace(",", ""))
                extracted_values.append((key.strip(), numeric_value))

    # Compute total sum of absolute values
    total_abs_value = sum(abs(value) for _, value in extracted_values)

    # Compute threshold in absolute value
    threshold_value = (threshold_percent / 100) * total_abs_value

    # Filter out values below the threshold
    for key, value in extracted_values:
        if abs(value) >= threshold_value:
            parsed_dict[key] = value

    return parsed_dict



################################################################
#### START PROGRAM ####
################################################################

# Function to open the app in the browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Read SHAP values from file
if os.path.exists(file_path):
    print("Path SHAP file exists: " + file_path)
    df = pd.read_csv(file_path, index_col=0)

    #delete columns that contain "Unnamed" in columns names
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    #delete columns that contain "Bias" in columns names
    df = df.loc[:, ~df.columns.str.contains('BiasTerm')]

    # Impute values in df
    df = df.fillna(df.mean())
    #delete columns that contain "Unnamed" in columns names
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    #delete columns that contain "APR Severity of Illness Description" in columns names
    df = df.loc[:, ~df.columns.str.contains('APR Severity of Illness Description')]

    # Impute values in df
    df = df.fillna(df.mean())
else:
    print("File not found")

# Read SHAP values from file
if os.path.exists(file_path_db):
    print("Path DB file exists: " + file_path_db)
    df_db = pd.read_csv(file_path_db, index_col=0)

     #delete columns that contain "APR Severity of Illness Description" in columns names
    df_db = df_db.loc[:, ~df_db.columns.str.contains('APR Severity of Illness Description')]

    df_db = df_db.sort_index()
    df_db['Total Costs'] = np.log1p(df_db['Total Costs'])

    df_index = df_db.index

else:
    print("DB not found: ", file_path_db)

# Read input data from file
if os.path.exists(inputdata_path_db):
    print("Path Input DB file exists: " + inputdata_path_db)
    df_input_db = pd.read_csv(inputdata_path_db, index_col=0)
    # drop column 'Total Costs' from input data
    #df_input_db = df_input_db.drop('Unnamed: 0', axis=1)
    df_input_db = df_input_db.drop('Total Costs', axis=1)

    # drop all rows in df_input_db with index not in df_db
    df_input_db = df_input_db[df_input_db.index.isin(df_db.index)] 

else:
    print("DB not found: ", file_path_db)

# Read SHAP values from file
if os.path.exists(original_path_db):
    print("Path DB file exists: " + original_path_db)
    #fn_db = pd.read_csv(original_path_db, index_col=0, usecols=['Facility Name'])
    fn_db = pd.read_csv(original_path_db, index_col=0)

    drop_cols = ['Birth Weight', 'Abortion Edit Indicator'] #, 'Total Costs']
    fn_db = fn_db.drop(drop_cols, axis=1)

    # drop all columns in fn_db except 'Facility Name'
    #fn_db = fn_db.drop(fn_db.columns.difference(['Facility Name']), axis=1)

    fn_db = fn_db.loc[fn_db.index.isin(df_db.index)]

    # fn_list is feature 'Facility Name' in fn_db
    fn_list = ['All'] + list(set(fn_db['Facility Name']))

else:
    print("DB not found: ", file_path_db)

# load model
# model_path = './models/' + app_name + '/model'
# model = joblib.load(model_path)

# load h2o gbm model
model = h2o.load_model(gbm_model_path)

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1(
        "AI-Model Evaluation Cockpit",
        style={
            'font-family': 'Arial',
            'background-color': 'black',
            'color': 'white',
            'padding': '10px',
            'text-align': 'center',
            'position': 'fixed',  # Fix the header to the top
            'top': '0',  # Stick it to the top
            'left': '0',  # Align to left
            'width': '100%',  # Full width
            'z-index': '1000'  # Ensure it stays above other elements
        }
    ),

    # Add margin to prevent content from overlapping the fixed header
    html.Div(style={'marginTop': '100px'}), # Adjust margin based on header height
    #html.Button('Close', id='close-button', n_clicks=0),

    #html.Br(),
    #html.Br(),

    #html.Label("Select an Item:"),  # Title for the dropdown
        # "Select Case" and Dropdown in the same row
    html.Div([
        html.H2(
            "Select Case",
            style={
                'font-family': 'Arial',
                'color': 'blue',
                'marginRight': '15px',  # Spacing between text and dropdown
                'whiteSpace': 'nowrap'  # Prevents line break
            }
        ),

        dcc.Dropdown(
            id='item-dropdown',
            options=[{'label': index, 'value': index} for index in df_index],
            value=df_index,
            style={
                'font-family': 'Arial',
                'width': '5cm'
            }
        )
    ], style={
        'display': 'flex',  # Flexbox to align items in one row
        'alignItems': 'center',  # Vertically align text & dropdown
        'gap': '10px'  # Add spacing between elements
    }),
    
        # Loading spinner wrapping the components
    dcc.Loading(
        id="loading-icon",
        type="default",  # You can change the spinner type: 'graph', 'default', etc.
        children=[
        html.Br(),

        html.Div([
            # Firework Plot Section (Left)
            html.Div([
                html.H3("Feature Signal Strength", style={
                    'textAlign': 'center',
                    'fontFamily': 'sans-serif',
                    'fontSize': '16pt',
                    'color': 'blue'
                }),
                html.H4("AI-Model Firework Plots", style={
                    'textAlign': 'center',
                    'fontFamily': 'sans-serif',
                    'fontSize': '14pt',
                    'color': 'blue'
                }),
                dcc.Graph(id='combined-firework-plot', style={'width': '100%', 'height': '800px'})  # Adjusted size
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '30%', 'marginRight': '10px'}),  

            # Middle Section: Radar Plot & SHAP Table
            html.Div([
                # Radar Plot
                html.Div([
                    html.H3("SHAP Radar Plot", style={
                        'textAlign': 'center',
                        'fontFamily': 'sans-serif',
                        'fontSize': '16pt',
                        'color': 'blue'
                    }),
                    html.H4("AI Model of Selected Sample", style={
                        'textAlign': 'center',
                        'fontFamily': 'sans-serif',
                        'fontSize': '14pt',
                        'color': 'blue'
                    }),
                    dcc.Graph(id='radar-plot', style={'width': '80%', 'height': 'auto'})  # Reduced height
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100%'}),  

                # SHAP Table - Now properly below the Radar Plot
                html.Div([
                    html.H4("SHAP Value Contributions", style={
                        'textAlign': 'center',
                        'fontFamily': 'sans-serif',
                        'fontSize': '14pt',
                        'color': 'blue',
                        'marginTop': '20px'
                    }),
                    html.Div(id='shap-table')  # Dynamically updated SHAP table
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100%', 'marginTop': '10px'})  
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '40%'}),  

            # Right Section: Ground Truth Data Table
            html.Div([
                html.H3("Ground Truth of Selected Sample", style={
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'fontFamily': 'sans-serif',
                    'fontSize': '16pt',
                    'color': 'blue'
                }),

                dbc.Button(
                    "Show/Hide Table",
                    id="toggle-collapse",
                    color="primary",
                    n_clicks=0,
                    style={'marginBottom': '10px', 'fontSize': '12pt', 'padding': '10px'}
                ),

                dbc.Collapse(
                    id="collapse",
                    is_open=False,
                    children=[
                        html.Div(
                            id='data-table',
                            children="This is your data table content here.",
                            style={
                                'border': '1px solid black',
                                'padding': '10px',
                                'width': '100%'
                            }
                        )
                    ]
                )
            ], style={'width': '30%', 'padding-left': '10px'})  

        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'width': '100%'}),  # Main container with flex layout

        html.Br(),

        # Sliders for y_cut and combined firework x-axis range, placed side by side
        html.Div([
            # Slider for y_cut
            html.Div([
                #html.Label("Adjust Y Cut-off:"),
                html.H2("Adjust Y Cut-off", style={'font-family': 'Arial', 'color': 'blue'}),
                dcc.Slider(
                    id='y-cut-slider',
                    min=0,
                    max=10,
                    step=0.2,
                    value=y_cut,  # Use a default value or pass y_cut if global
                    marks={i: str(i) for i in range(0, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '20%'}),  # Set width to 20%

            # Combined firework x-axis range slider
            html.Div([
                #html.Label("Combined Firework Plot X-axis Range:"),
                html.H2("Combined Firework Plot X-axis Range", style={'font-family': 'Arial', 'color': 'blue'}),
                dcc.RangeSlider(
                    id='combined-firework-x-range-slider',
                    min=-2000,
                    max=2000,
                    step=0.1,
                    value=[-1000, 1000],
                    marks={i: str(i) for i in range(-2000, 2000, 500)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '40%'})  # Set width to 40%
        ], style={'display': 'flex', 'align-items': 'center'}),  # Use flex to position them side by side

        html.Div(id='slider-output'),

        html.Br(),

        # Parent div to position the classification table and feature checklist side by side
        html.Div([
            # Classification table
            html.Div([
                html.H2("Divergence Table", style={'font-family': 'Arial', 'color': 'blue'}),
                dash_table.DataTable(
                    id='classification-table',
                    columns=[
                        {'name': 'Feature', 'id': 'Feature'},
                        {'name': 'Classification', 'id': 'Classification'},
                        {
                            'name': 'Divergence pos',
                            'id': 'DivPos',
                            'type': 'numeric',
                            'format': {'specifier': '.2f'}
                        },
                        {
                            'name': 'Divergence neg',
                            'id': 'DivNeg',
                            'type': 'numeric',
                            'format': {'specifier': '.2f'}
                        },
                    ],
                    style_table={'width': '100%', 'margin-top': '20px'},
                    style_cell={'font-family': 'Arial','textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},

                    # Conditional styling to make values < 1.0 bold
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{DivPos} < 1 && {DivPos} > 0',
                                'column_id': 'DivPos'
                            },
                            'fontWeight': 'bold',
                            'color': 'blue'
                        },
                        {
                            'if': {
                                'filter_query': '{DivNeg} < 1 && {DivNeg} > 0',
                                'column_id': 'DivNeg'
                            },
                            'fontWeight': 'bold',
                            'color': 'blue'
                        }
                    ]
                )
            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '10px'}),  # Adjust width to match feature checklist

            # Feature checklist and buttons to the right of the classification table
            html.Div([
                html.H2("Select Features for Combined Firework Plot", style={'font-family': 'Arial', 'color': 'blue'}),
                html.Br(),

                # Feature checklist allowing multiple selections
                dcc.Checklist(
                    id='feature-checklist',
                    options=[{'label': feature, 'value': feature} for feature in df.columns],
                    value=[],  # Default: no features selected
                    inline=False,
                    style={'font-family': 'Arial'}
                ),
                html.Br(),

                # Buttons for selecting/deselecting features
                html.Button("Select All", id="select-all", n_clicks=0),
                html.Button("Deselect All", id="deselect-all", n_clicks=0),
                html.Button("Select Blue Features", id="select-blue", n_clicks=0),
                
                html.Br(),
                html.Button("Update Plot", id="update-plot", n_clicks=0, style={'margin-top': '10px', 'font-weight': 'bold'}),  # New update button

            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '30px'})

        ], style={'display': 'flex', 'align-items': 'flex-start'}),  # Flex to place elements side by side

        html.Br(),

        # Parent div to place the dropdown and slider side by side
        html.Div([
            html.Div([
                html.H2("Select Facility", style={'font-family': 'Arial', 'color': 'blue'}),
                dcc.Dropdown(
                    id='facility-dropdown',
                    options=[{'label': feature, 'value': feature} for feature in fn_list],
                    value="Mount Sinai Hospital",  # Set default value to "Mount Sinai Hospital"
                    #value=df.columns[0],
                    style={'width': '100%'}  # Full width within its container
                )
            ], style={'font-family': 'Arial','width': '20%', 'padding-right': '20px'}),  # Adjust width and spacing

            # Feature dropdown
            html.Div([
                html.H2("Select Features", style={'font-family': 'Arial', 'color': 'blue'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': feature} for feature in df.columns],
                    value="CCS Procedure Description",
                    #value=df.columns[0],
                    style={'font-family': 'Arial','width': '100%'}  # Full width within its container
                )
            ], style={'width': '20%', 'padding-right': '20px'}),  # Adjust width and spacing

        ], style={'display': 'flex', 'align-items': 'top'}),  # Use flexbox to place elements side by side

        html.Br(),
        
        html.Div([
            # Histogram Plot
            dcc.Graph(id='histogram-plot', style={'width': '50%'}),

                        # Feature value from selected feature
            html.Div([
                html.H2("Select Value from Selected Feature", style={'font-family': 'Arial', 'color': 'blue'}),
                dcc.Dropdown(
                    id='feature-value-dropdown',
                    placeholder="Select Feature Value",
                    style={'font-family': 'Arial','width': '100%'}  # Full width within its container
                ),
                dcc.Graph(id='density-plot')  # Graph component for density plot,
            ], style={'width': '50%', 'padding-right': '60px'}),  # Adjust width and spacing
        ], style={'display': 'flex', 'align-items': 'flex-start', 'gap': '20px', 'width': '100%'}),

        html.Br(),

        html.Div([
            # Selected Facility Table with Title
            html.Div([
                html.H2("Selected Facility", style={'font-family': 'Arial', 'color': 'red', 'textAlign': 'center'}),
                dash_table.DataTable(
                    id='result-table',
                    columns=[
                        {'name': 'Most Frequent Value', 'id': 'Most Frequent Value', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Feature Value', 'id': 'Feature Value'},
                        {'name': '#', 'id': 'Frequency'}
                    ],
                    data=[],
                    style_table={'width': '100%', 'margin-top': '10px'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                )
            ], style={'width': '33%', 'padding': '10px'}),  # Set equal width and padding for spacing

            # All Facilities Table with Title
            html.Div([
                html.H2("All Facilities", style={'font-family': 'Arial', 'color': 'blue', 'textAlign': 'center'}),
                dash_table.DataTable(
                    id='result-all-table',
                    columns=[
                        {'name': 'Most Frequent Value', 'id': 'Most Frequent Value', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Feature Value', 'id': 'Feature Value'},
                        {'name': '#', 'id': 'Frequency'}
                    ],
                    data=[],
                    style_table={'width': '100%', 'margin-top': '10px'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                )
            ], style={'width': '33%', 'padding': '10px'})  # Set equal width and padding for spacing
        ], style={'display': 'flex', 'align-items': 'flex-start', 'gap': '20px', 'width': '100%'}),

        html.Br(),

        html.Div([
            # Prominence slider
            html.Div([
                #html.Label("Adjust Prominence for Peak Detection:"),
                html.H2("Adjust Prominence for Peak Detection", style={'font-family': 'Arial', 'color': 'blue', 'textAlign': 'center'}),
                dcc.Slider(
                    id='prominence-slider',
                    min=0.0,
                    max=0.00001,
                    step=0.000001,
                    value=0.000001,
                    marks={i/10: str(i/10) for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%'}),  # Adjust width

            # Density plot x-axis range slider
            html.Div([
                #html.Label("Adjust Density Plot X-axis Range:", style={'font-size': '14px', 'font-weight': 'bold'}),
                html.H2("Adjust Density Plot X-axis Range", style={'font-family': 'Arial', 'color': 'blue', 'textAlign': 'center'}),
                dcc.RangeSlider(
                    id='density-x-range-slider',
                    min=-20000,
                    max=20000,
                    step=0.01,
                    value=[-15000, 25000],
                    marks={i: {'label': str(i), 'style': {'color': 'red' if i == 0 else 'black'}} for i in range(-15000, 15001, 2500)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%'}),  # Adjust width
            html.Div([
                #html.Label("Adjust Density Plot X-axis Range:", style={'font-size': '14px', 'font-weight': 'bold'}),
                html.H2("Peak Epsilon", style={'font-family': 'Arial', 'color': 'blue', 'textAlign': 'center'}),
                dcc.Slider(
                    id='peak-epsilon',
                    min=0,
                    max=20,
                    step=0.5,
                    value=10,  # Default starting value
                    marks={i: {'label': str(i), 'style': {'color': 'red' if i == 0 else 'black'}} for i in range(0, 21, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%'})  # Adjust width
        ], style={'display': 'flex', 'align-items': 'center'})  # Use flexbox to place elements side by side
    ]),
    # Footer with attribution
    html.Div(
        children="Conceptual Alignment Method App built by Wolfgang Maass, Saarland University ©2024",
        style={
            'position': 'fixed',
            'bottom': '0',
            'width': '100%',
            'backgroundColor': 'black',
            'color': 'white',
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Arial',
            'fontSize': '12px',
            'fontStyle': 'italic'
        }
    ),
    html.Br(),
    html.Br()
])

# Callback to toggle the collapse
@app.callback(
    Output("collapse", "is_open"),  # Output: toggling collapse
    [Input("toggle-collapse", "n_clicks")],  # Input: button click
    [dash.dependencies.State("collapse", "is_open")]  # State: current collapse status
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:  # Check if button was clicked
        return not is_open  # Toggle the state
    return is_open  # Return the current state if no click


# Callback to update the individual plots and combined firework plot
@app.callback(
    [Output('histogram-plot', 'figure'),
     Output('classification-table', 'data'),
     Output('combined-firework-plot', 'figure'),
     Output('slider-output', 'children'),
     Output('result-table', 'data'),
     Output('result-all-table', 'data'),
     Output('feature-value-dropdown', 'options')],
    [Input('facility-dropdown', 'value'),
     Input('feature-dropdown', 'value'), 
     Input('prominence-slider', 'value'),
     Input('y-cut-slider', 'value'),
     Input('combined-firework-x-range-slider', 'value'),
     Input('density-x-range-slider', 'value'),
     Input('peak-epsilon', 'value'),
     Input('update-plot', 'n_clicks')],  # Add explicit update trigger
    [State('feature-checklist', 'value')]  # Read feature selection without triggering updates
)
def update_plots(selected_facility, selected_feature, prominence, y_cut, 
                 combined_firework_x_range, density_x_range, peak_epsilon, 
                 update_clicks, selected_features):
    
    # Set a default facility if none is selected
    if not selected_facility:
        selected_facility = 'Mount Sinai Hospital'

    # Ensure at least one feature is selected
    if not selected_features:  
        selected_features = ['CCS Procedure Description']

    # Prepare the data (handle errors gracefully)
    try:
        firework_df, peak_classification, peaks_x = prepare_firework_data(df, selected_feature, prominence)
        all_classifications = compute_all_peak_classifications(df, prominence, y_cut)
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return {}, {}, [], {}, "Error", []  

    # Save all classifications to a JSON file
    file_path2 = f'./results/{app_name}/classifications.json'
    try:
        with open(file_path2, 'w') as json_file:
            json.dump(all_classifications, json_file, indent=4)
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

    # Compute SHAP values and log2 transformations
    try:
        shap_val = df[selected_feature] - df[selected_feature].mean()
        log2_y = np.log2(np.abs(df[selected_feature] / df[selected_feature].mean()))
        shap_val += np.random.normal(0, 1e-6, size=shap_val.shape)  
    except KeyError as e:
        print(f"KeyError: {e}")
        return {}, {}, [], {}, "Invalid Feature", []

    # Create the Firework Plot
    firework_fig = px.scatter(
        firework_df,
        x='shap_val',
        y='Log2_Y',
        title=f'Firework Plot of {selected_feature} - {peak_classification}',
        labels={'shap_val': 'SHAP - mean', 'Log2_Y': 'Log2(Impact)'},
        template='plotly_white'
    )

    firework_fig.update_traces(
        mode='markers',
        marker=dict(size=1, line=dict(width=1), opacity=0.8)
    )

    firework_fig.update_layout(
        height=900,
        xaxis_title='SHAP - mean',
        yaxis_title='Log2(Impact)',
        xaxis=dict(range=[-15000, 15000]),
        title={
            'text': f'Firework Plot of {selected_feature} - {peak_classification}',
            'font': {'size': 18, 'family': 'Arial', 'color': 'black', 'weight': 'bold'},
            'x': 0.5
        }
    )

    # Compute Density Plot and KDE
    try:
        density = gaussian_kde(shap_val)
        xs = np.linspace(min(shap_val), max(shap_val), 1000)
        density_values = density(xs)
        peaks, _ = find_peaks(density_values, prominence=prominence)

        if selected_facility is not None and selected_facility != 'All':
            facility_indexes = fn_db[fn_db['Facility Name'] == selected_facility].index
        else:
            facility_indexes = df_db.index  

        shap_val_filtered = df.loc[facility_indexes, selected_feature] - df[selected_feature].mean()
        df_filtered = df.loc[facility_indexes]

        density_filtered = gaussian_kde(shap_val_filtered)
        density_values_filtered = density_filtered(xs)
        peaks_filtered, _ = find_peaks(density_values_filtered, prominence=prominence)

    except Exception as e:
        print(f"Error in density calculation: {e}")
        return firework_fig, {}, [], {}, "Error", []

    most_frequent_values_all = xs[peaks]
    indexes_all_dict = collect_indexes_around_values(df, selected_feature, most_frequent_values_all, peak_epsilon)

    most_frequent_values = xs[peaks_filtered]
    indexes_dict = collect_indexes_around_values(df_filtered, selected_feature, most_frequent_values, peak_epsilon)

    # Create Density Plot
    density_fig = go.Figure()
    density_fig.add_trace(go.Scatter(
        x=xs, y=density_values, mode='lines',
        line=dict(color='black', width=2),
        name=wrap_text('All Facilities', 5)
    ))

    if selected_facility is not None and selected_facility != 'All':
        density_fig.add_trace(go.Scatter(
            x=xs, y=density_values_filtered, mode='lines',
            line=dict(color='red', width=2),
            name=wrap_text(f'KDE for {selected_facility}', 5)
        ))

    for peak in peaks:
        peak_x = xs[peak]
        peak_y = density_values[peak]
        density_fig.add_trace(go.Scatter(
            x=[peak_x], y=[peak_y],
            mode='markers+text', marker=dict(color='red', size=10),
            text=[f'{peak_x:.2f}'], textposition="top center", showlegend=False
        ))

    if selected_facility is not None and selected_facility != 'All':
        for peak in peaks_filtered:
            peak_x = xs[peak]
            peak_y = density_values_filtered[peak]
            density_fig.add_trace(go.Scatter(
                x=[peak_x], y=[peak_y],
                mode='markers+text', marker=dict(color='red', size=10),
                text=[f'{peak_x:.2f}'], textposition="top center", showlegend=False
            ))

    density_fig.update_layout(
        height=900,
        title={
            'text': f'Density Plot of KDE of SHAP values for {selected_feature}',
            'font': {'size': 24, 'family': 'Arial', 'color': 'blue', 'weight': 'bold'},
        },
        xaxis_title='SHAP - mean',
        yaxis_title='Density',
        template='plotly_white',
        xaxis=dict(
            range=density_x_range,
            rangeslider=dict(visible=True),
            rangemode='normal'
        )
    )

    # **Update Combined Firework Plot ONLY when Update Button is Pressed**
    combined_firework_fig = combined_firework_plot(df, prominence, selected_features, combined_firework_x_range, y_cut)

    slider_output = f"Y Cut-off value: {y_cut}"

    try:
        result_df = analyze_frequent_values(fn_db, selected_facility, selected_feature, indexes_dict)
        result_table = result_df.to_dict('records')

        result_all_df = analyze_frequent_values(fn_db, "All", selected_feature, indexes_all_dict)
        result_all_table = result_all_df.to_dict('records')

        feature_value_options = [{'label': value, 'value': value} for value in result_df['Feature Value'].unique()]

    except Exception as e:
        print(f"Error in generating result table: {e}")
        result_table = []

    return density_fig, all_classifications, combined_firework_fig, slider_output, result_table, result_all_table, feature_value_options



#--------------------------------------------------------------------

# Radar plot update based on clickData in the combined firework plot
@app.callback(
    [Output('radar-plot', 'figure'),
     Output('data-table', 'children'),
     Output('shap-table', 'children')],  # Update both radar plot and data table
    [Input('combined-firework-plot', 'clickData'),
     Input('classification-table', 'data'),
     Input('item-dropdown', 'value')]  # Include classification table data as input
)
def display_radar_plot(clickData, table_data, selected_item):

    if selected_item is not None:
        point_index = selected_item
    elif clickData is None:
        clickData = {'points': [{'curveNumber': 0, 'pointNumber': 39866, 'pointIndex': 39866, 'x': 912.780153087414, 'y': 1.2089790007612493, 'bbox': {'x0': 558.61, 'x1': 561.61, 'y0': 437.76, 'y1': 440.76}, 'customdata': [1539665, -394.84554760732857]}]}
        point_index = clickData['points'][0]['customdata'][0]
    else:
        point_index = clickData['points'][0]['customdata'][0]

    # if not clickData:
    #     # Return an empty radar plot and empty table if nothing is clicked
    #     return go.Figure(), html.Div()

    # Extract the index of the clicked data point
    #point_index = clickData['points'][0]['pointIndex']

    # Ensure the point_index exists in both df and df_db
    if point_index not in df.index or point_index not in df_db.index:
        return go.Figure(), html.Div()

    # Extract corresponding data for the radar plot from df
    selected_sample = df.loc[point_index]

    # Extract corresponding data for the table from f_db
    selected_sample_db = fn_db.loc[point_index]
    total_cost_real = selected_sample_db['Total Costs']

    # Extract corresponding data for the inputdata_path_db
    selected_sample_input = df_input_db.loc[point_index]

    # Extract features marked in blue from the classification table
    blue_features = set()  # Use a set to hold the blue-marked features
    for row in table_data:
        if 0 < row['DivPos'] < 1 or 0 < row['DivNeg'] < 1:
            blue_features.add(row['Feature'])

    # Separate positive and negative values for the radar plot
    positive_values = [val if val > 0 else 0 for val in selected_sample.values]
    negative_values = [val if val < 0 else 0 for val in selected_sample.values]

    # Create hover text that only shows for non-zero values
    hovertext_positive = [f'{feature}: ${val:,.2f}' if val != 0 else '' for feature, val in zip(selected_sample.index, positive_values)]
    hovertext_negative = [f'{feature}: ${val:,.2f}' if val != 0 else '' for feature, val in zip(selected_sample.index, negative_values)]

    # Merge hovertext_positive and hovertext_negative
    hovertext_combined = hovertext_positive + hovertext_negative

    # Apply function with a threshold of 5%
    parsed_data = parse_hovertext(hovertext_combined, threshold_percent=3)
    # Convert to DataFrame and display
    df_shap = pd.DataFrame(list(parsed_data.items()), columns=["Category", "Amount (USD)"])

    print(df_shap)

    # Create SHAP table
    shap_table = dash_table.DataTable(
        id='shap-values-table',
        columns=[{"name": i, "id": i} for i in df_shap.columns],
        data=df_shap.to_dict('records'),
        style_table={'width': '80%', 'margin': 'auto'},
        style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center', 'fontSize': '12pt'}
    )

    # Create a radar plot
    radar_fig = go.Figure()

    # Add positive values with red fill and hover information
    radar_fig.add_trace(go.Scatterpolar(
        r=positive_values,
        theta=selected_sample.index,
        fill='toself',
        name='Positive Values',
        fillcolor='rgba(78, 121, 255, 1)',  # Red with some transparency
        
        line_color='blue',
        mode='lines+markers',  # Show both lines and markers
        marker=dict(
            size=6,
            color='blue',  # Red markers for positive values
            symbol='circle'
        ),
        hoverinfo='text',
        hovertext=hovertext_positive
    ))

    # Define circle points at value 0
    theta_circle = selected_sample.index
    r_circle = [0] * len(selected_sample.index)

    # Add zero circle trace
    radar_fig.add_trace(go.Scatterpolar(
        r=r_circle,
        theta=theta_circle,
        mode='lines',
        name='Zero Circle',
        line=dict(color='black', width=2, dash='dash'),
        fill='toself',
        fillcolor='rgba(255, 78, 78, 1)',  # Neutral grey for zero
        showlegend=False
    ))

    # Add negative values with blue fill and hover information
    radar_fig.add_trace(go.Scatterpolar(
        r=negative_values,
        theta=selected_sample.index,
        fill='toself',
        name='Negative Values',
        fillcolor='rgba(230, 236, 245, 1)',  # Blue with some transparency
        #fillcolor='white',  # Blue with some transparency
        line_color='red',
        mode='lines+markers',
        marker=dict(
            size=6,
            color='red',  # Blue markers for negative values
            symbol='circle'
        ),
        hoverinfo='text',
        hovertext=hovertext_negative
    ))

    # Update the radar plot layout
    radar_fig.update_layout(
        title={
            'text': f'Selected Sample: {point_index} <br>',  # Title text
            'x': 0.5,  # Centers the title
            'xanchor': 'center',  # Ensures the anchor is centered
            'yanchor': 'top'  # Keeps the vertical anchor at the top
        },
        polar=dict(
            radialaxis=dict(visible=True, range=[min(selected_sample.values), max(selected_sample.values)]),
            angularaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12, color='black')  # Global color for all labels
            )
        ),
        margin=dict(l=200, r=10, t=25, b=50),
        height=900,
        showlegend=True
    )

    # Add blue dashed lines for specific features
    for feature in blue_features:
        if feature in selected_sample.index:
            radar_fig.add_trace(go.Scatterpolar(
                r=[0, max(selected_sample.values)],  # Line from center to maximum value
                theta=[feature, feature],
                mode='lines',
                line=dict(color='rgba(78, 121, 255, 1)', width=1, dash='dash'),
                showlegend=False
            ))

    # Add annotations for blue-marked features
    for feature in blue_features:
        if feature in selected_sample.index:
            radar_fig.add_annotation(
                text=f'<b>{feature}</b>',
                xref='paper',
                yref='paper',
                x=0.5,
                y=1.1,
                showarrow=False,
                font=dict(size=18, color='rgba(78, 121, 255, 1)')
            )


    # for HDMA data because log(Total Costs) is used for training
    if app_name == 'HIS17-two-classes':
        selected_sample_db['Total Costs'] = np.expm1(selected_sample_db['Total Costs'])
        # format selected_sample_db['Total Costs'] float with two digits
        selected_sample_db['Total Costs'] = selected_sample_db['Total Costs'].round(2)

    # create dataframe sample_df with column names df_db.columns and data selected_sample_db
    #sample_df = pd.DataFrame(columns=fn_db.columns, data=[selected_sample_db.values])
    
    #sample_df = selected_sample.to_frame().T
    sample_df = df_db.loc[point_index].to_frame().T

    columns = sample_df.columns

    # Identify columns with and without 'encoded_' prefix
    encoded_columns = [col for col in columns if col.startswith('encoded_')]
    non_encoded_columns = [col for col in columns if not col.startswith('encoded_')]

    # Find non-encoded columns that have an encoded counterpart
    to_drop = [col for col in non_encoded_columns if f'encoded_{col}' in encoded_columns]

    # Drop the non-encoded columns that have an encoded counterpart
    sample_df = sample_df.drop(columns=to_drop)

    #drop_cols = ['Total Costs']

    # Drop columns from sample_df
    #sample_df.drop(columns=drop_cols, inplace=True)

    

    sample_h20 = h2o.H2OFrame(sample_df)

    # Use the model to make a prediction
    prediction = model.predict(sample_h20)

    p = prediction.as_data_frame()
    p1 = p.iloc[0,0]
    p1f = round(p1, 2)

    selected_sample_db[target] = total_cost_real

    delta = round(np.abs(p1f - total_cost_real), 2)

    # Create a vertically organized data table
    table = dash_table.DataTable(
        columns=[{'name': 'Feature', 'id': 'Feature'}, {'name': 'Value', 'id': 'Value'}],
        data=[{'Feature': feature, 'Value': selected_sample_db[feature]} for feature in selected_sample_db.index] 
        + [{'Feature': 'Prediction', 'Value': p1f}]
        + [{'Feature': 'Delta', 'Value': delta}],
        #+ [{'Feature': 'Real Value', 'Value': total_cost_real}],
        style_table={'width': '100%', 'margin-top': '20px'},
        style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal'},  # Wrap long text
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},

        # Conditional styling for Prediction and Delta rows
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Feature} = "Prediction"'  # Applies to the Prediction row
                },
                'fontWeight': 'bold'  # Makes the row bold
            },
            {
                'if': {
                    'filter_query': '{Feature} = "Delta"'  # Applies to the Delta row
                },
                'fontWeight': 'bold'  # Makes the row bold
            },
            {
                'if': {
                    'column_id': 'Value',  # Ensure only the Value column for Prediction and Delta is bold
                    'filter_query': '{Feature} = "Prediction" || {Feature} = "Delta"'
                },
                'fontWeight': 'bold'  # Set font weight to bold for Value column in those rows
            }
        ]
    )

    return radar_fig, table, shap_table

# Callback to keep the range slider synchronous
@app.callback(
    Output('combined-firework-x-range-slider', 'value'),
    [Input('combined-firework-x-range-slider', 'value')]
)
def synchronize_slider(value):
    # The midpoint should always be the center, here it's 0
    midpoint = 0

    # Calculate the distance from the midpoint to each side of the range
    left_distance = abs(midpoint - value[0])
    right_distance = abs(midpoint - value[1])

    # If left side is adjusted, adjust the right side symmetrically
    if left_distance != right_distance:
        return [-left_distance, left_distance]

    return value

@app.callback(
    Output('feature-checklist', 'value'),  # Update the selected features in the checklist
    [Input('select-all', 'n_clicks'),      # Button to select all features
     Input('deselect-all', 'n_clicks'),    # Button to deselect all features
     Input('select-blue', 'n_clicks')],    # Button to select blue-marked features
    [State('feature-checklist', 'options'),  # Access the checklist options
     State('classification-table', 'data')], # Access the data from the classification table
    prevent_initial_call=True
)
def update_feature_checklist(select_all_clicks, deselect_all_clicks, select_blue_clicks, options, table_data):
    # Get the context to see which input triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Identify which button triggered the callback
    triggered_button = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_button == 'select-all':
        # Return all features if "Select All" was clicked
        return [option['value'] for option in options]  # Select all options

    elif triggered_button == 'deselect-all':
        # Return an empty list if "Deselect All" was clicked
        return []

    elif triggered_button == 'select-blue':
        # Return only blue-marked features (those with DivPos < 1 or DivNeg < 1)
        selected_features = []
        if table_data:  # Check if the table data is available
            for row in table_data:
                if ('DivPos' in row and 0 < row['DivPos'] < 1) or ('DivNeg' in row and 0 < row['DivNeg'] < 1):
                    selected_features.append(row['Feature'])  # Add feature to the selected list
        return selected_features

    return dash.no_update  # In case no action is required

# Callback to filter df based on selected feature value and plot density with a red dot for the selected facility
@app.callback(
    Output('density-plot', 'figure'),
    [Input('feature-value-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('facility-dropdown', 'value')]  # `facility-dropdown` for selecting the facility
)
def filter_and_plot_density(selected_value, selected_feature, selected_facility):
    # Return an empty figure if inputs are None
    if selected_value is None or selected_feature is None or selected_facility is None:
        return go.Figure()

    # Filter fn_db for rows where the selected feature has the selected value and the facility name matches
    filtered_indexes_facility = fn_db[(fn_db[selected_feature] == selected_value) & 
                                      (fn_db['Facility Name'] == selected_facility)].index
    filtered_values_facility = df.loc[filtered_indexes_facility, selected_feature]

    # plot histogram filtered_values_facility and save png
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=filtered_values_facility, histnorm='probability', name=selected_facility))
    fig.write_image('facility.png')

    print(filtered_values_facility.mean())

    # Filter fn_db for rows where the selected feature has the selected value across all facilities
    filtered_indexes_all = fn_db[fn_db[selected_feature] == selected_value].index

    # delete filtered_indexes_facility from filtered_indexes_all
    filtered_indexes_all = filtered_indexes_all.drop(filtered_indexes_facility)

    filtered_values_all = df.loc[filtered_indexes_all, selected_feature]

    # plot histogram filtered_values_facility and save png
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=filtered_values_all, histnorm='probability', name="All"))
    fig.write_image('all_facility.png')

    print(filtered_values_all.mean())

    ################################################################

    # Set a default bandwidth if none is provided
    bandwidth = 'scott'  # Default to 'scott' if not specified

    # Check if there are any values to plot for the selected facility
    if len(filtered_values_facility) == 0 or len(filtered_values_all) == 0:
        return go.Figure()  # Return an empty figure if no data is found

    # Convert to Pandas Series for easier handling
    filtered_values_facility = pd.Series(filtered_values_facility)
    filtered_values_all = pd.Series(filtered_values_all)

    # Convert to numeric, coercing errors to NaN
    filtered_values_facility = pd.to_numeric(filtered_values_facility, errors='coerce')
    filtered_values_all = pd.to_numeric(filtered_values_all, errors='coerce')

    # Remove NaN values
    filtered_values_facility = filtered_values_facility.dropna()
    filtered_values_all = filtered_values_all.dropna()

    # Convert back to NumPy arrays
    filtered_values_facility = filtered_values_facility.to_numpy()
    filtered_values_all = filtered_values_all.to_numpy()

    # Check if the arrays are empty after cleaning
    if filtered_values_facility.size == 0 or filtered_values_all.size == 0:
        return go.Figure()  # Return an empty figure if no valid data remains

    # Generate density estimations
    density_facility = gaussian_kde(filtered_values_facility, bw_method=bandwidth)
    density_all = gaussian_kde(filtered_values_all, bw_method=bandwidth)

    # Dynamically compute the range to capture density tails
    data_min = min(filtered_values_facility.min(), filtered_values_all.min())
    data_max = max(filtered_values_facility.max(), filtered_values_all.max())
    range_extension = (data_max - data_min) * 0.5  # Extend by 50% of the data range

    # Create extended range for x values
    x_vals = np.linspace(
        data_min - range_extension,
        data_max + range_extension,
        5000  # Higher resolution to capture finer details
    )

    # Evaluate densities
    y_vals_facility = density_facility(x_vals)
    y_vals_all = density_all(x_vals)

    # Normalize y-values for better comparison
    y_vals_facility /= y_vals_facility.max()
    y_vals_all /= y_vals_all.max()

    # Dynamically filter for non-zero density values to avoid clipping
    significant_density_threshold = 1e-3  # A small threshold for tail cutoff
    x_vals_facility = x_vals[y_vals_facility > significant_density_threshold]
    y_vals_facility = y_vals_facility[y_vals_facility > significant_density_threshold]
    x_vals_all = x_vals[y_vals_all > significant_density_threshold]
    y_vals_all = y_vals_all[y_vals_all > significant_density_threshold]

    # Create Plotly figure for both density plots
    fig = go.Figure()

    # Add density plot for the selected facility
    fig.add_trace(go.Scatter(
        x=x_vals_facility,
        y=y_vals_facility,
        mode='lines',
        name='Selected Facility',
        line=dict(color='red')  # Set the color to black
    ))

    # Add density plot for all facilities
    fig.add_trace(go.Scatter(
        x=x_vals_all,
        y=y_vals_all,
        mode='lines',
        name='All Facilities',
        line=dict(color='black')  # Set the color to black
    ))


    # Update layout for the figure
    fig.update_layout(
        title='Density Plot of Selected Feature Value',
        xaxis_title='Feature Value',
        yaxis_title='Density',
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8055, debug=True)


# Example usage (ensure the 'classification.json' file is in the correct path):
del_features = read_and_collect_features('./results/'+app_name+'/classifications.json', 1.0)

# Drop the selected columns from the DataFrame
df = df.drop(columns=del_features, errors='ignore')  # 'errors=ignore' to avoid errors if the column doesn't exist
df_input_db=df_input_db.drop(columns=del_features, errors='ignore')  # 'errors=ignore' to avoid errors if the column doesn't exist