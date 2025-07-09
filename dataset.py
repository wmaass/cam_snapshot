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

result_path = './results/'

original_path_db = './results/'+app_name+'/ranked/df_original.csv'

# Read SHAP values from file
if os.path.exists(original_path_db):
    print("Path DB file exists: " + original_path_db)
    #fn_db = pd.read_csv(original_path_db, index_col=0, usecols=['Facility Name'])
    fn_db = pd.read_csv(original_path_db, index_col=0)

# Count the number of samples for each unique item in the "Facility Name" column
facility_counts = fn_db['Facility Name'].value_counts().reset_index()

# Rename columns for readability
facility_counts.columns = ['Facility Name', 'Number of Samples']

# Sort the table by the number of samples in descending order
facility_counts = facility_counts.sort_values(by='Number of Samples', ascending=False)

# Save the sorted table to a CSV file
output_path = './results/facility_sample_counts.csv'
facility_counts.to_csv(output_path, index=False)

print(f"Facility sample counts saved to {output_path}")