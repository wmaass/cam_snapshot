import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
import mplcursors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

app_name = "NYHealthSPARC"

# Attempt to read the CSV file from the specified path
file_path = './results/'+app_name+'/shap_mlsmall.csv'

# Create output directory if it doesn't exist
figure_path = './figures/'+app_name
os.makedirs(figure_path, exist_ok=True)

# Create output directory if it doesn't exist
dist_path = './figures/'+ app_name + '/distributions'
os.makedirs(dist_path, exist_ok=True)

# Check if the file exists at the specified path
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # delete first column
    df = df.drop(df.columns[0], axis=1)

    # Display the first few rows of the DataFrame
    print(df.head())
else:
    print("File not found")


# Standardize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df)

df_scaled_data = pd.DataFrame(scaled_data, columns=df.columns)

################################################################

# Calculate the variance of each feature
feature_variances = df_scaled_data.var()
# Set epsilon value for variance threshold
epsilon = 0.0

# Identify features with variance smaller than epsilon
low_variance_features = feature_variances[feature_variances < epsilon].index.tolist()

# Remove low variance features from the DataFrame
df_filtered = df_scaled_data.drop(columns=low_variance_features)
scaled_data_filtered = scaler.fit_transform(df_filtered)
df_scaled_data_filtered = pd.DataFrame(scaled_data_filtered, columns=df_filtered.columns)

########################################################################

# Calculate the average SHAP value for each feature
average_shap_values = df_scaled_data_filtered.mean()

# Sort the average SHAP values for better visualization
average_shap_values = average_shap_values.sort_values(ascending=False)

# # Calculate the average of scaled_data for each feature
feature_averages = df_scaled_data_filtered.mean(axis=0)
#feature_averages = df.mean(axis=0)

num_columns = df_scaled_data_filtered.shape[1]

# Perform hierarchical clustering on columns
linked = linkage(df_scaled_data_filtered.T, 'ward') #complete
#linked = linkage(df_log.T, 'complete') #complete

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
cluster_range = range(2, 10)  # Testing different numbers of clusters
for n_clusters in cluster_range:
    clusters = fcluster(linked, n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(df_scaled_data_filtered.T, clusters)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters with the highest silhouette score
optimal_num_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]

opt_num_clusters = 5 # set to it because it works for the NY Healthcare dataset

# Use the optimal number of clusters found
clusters = fcluster(linked, opt_num_clusters, criterion='maxclust')

# Create a DataFrame to display the clusters and their corresponding features
clustered_columns = pd.DataFrame({'Feature': df_scaled_data_filtered.columns, 'Cluster': clusters})

# Calculate the strength of association for each feature with each cluster
cluster_strengths = pd.DataFrame(index=clustered_columns['Feature'].unique(), columns=[f'Cluster {i+1}' for i in range(opt_num_clusters)])

cluster_weight = {}
for cluster in range(1, opt_num_clusters + 1):
    cluster_features = clustered_columns[clustered_columns['Cluster'] == cluster]['Feature']
    cluster_data = df_scaled_data_filtered[cluster_features]
    cluster_means = cluster_data.mean()
    cluster_strengths.loc[cluster_features, f'Cluster {cluster}'] = cluster_means
    cluster_weight[cluster] = cluster_means.mean()

# Normalize the strengths
cluster_strengths = cluster_strengths.fillna(0)
cluster_strengths_normalized = cluster_strengths.div(cluster_strengths.sum(axis=1), axis=0)

# Identify valid features where at least one value in each feature's values is greater than epsilon
valid_features = []

for feature in cluster_strengths.index:
    if any(abs(cluster_strengths.loc[feature]) > epsilon):
        valid_features.append(feature)

# Filter 'clustered_columns' to keep only valid features
clustered_columns = clustered_columns[clustered_columns['Feature'].isin(valid_features)]

feature_avg_dict = dict(zip(df_scaled_data_filtered.columns, feature_averages))

# Extract values and apply Min-Max scaling
values = list(feature_avg_dict.values())
min_val = min(values)
max_val = max(values)

# Scale the values using Min-Max scaling
feature_avg_scaled_dict = {key: (value - min_val) / (max_val - min_val) for key, value in feature_avg_dict.items()}

# Order clusters by name
clusters_sorted = sorted(clustered_columns['Cluster'].unique(), key=lambda x: f"Cluster {x}")
print("clusters_sorted: ", clusters_sorted)

# ----- PLOTS


# Draw the graph using networkx
G = nx.DiGraph()

# Add nodes for each cluster
for cluster in clustered_columns['Cluster'].unique():
    cluster_features = clustered_columns[clustered_columns['Cluster'] == cluster]['Feature'].tolist()
    cluster_name = f'Cluster {cluster}'
    G.add_node(cluster_name)

# Add a node for "total costs"
G.add_node('total costs')

# Add edges from each cluster to "total costs" with edge labels from cluster_avg_dict
edge_labels = {}
for cluster in clusters_sorted:
    cluster_name = f'Cluster {cluster}'
    G.add_edge(cluster_name, 'total costs')
    edge_labels[(cluster_name, 'total costs')] = f"{cluster_weight[cluster]:.2f}"


# # Add edges from each cluster to "total costs"
# for cluster in clustered_columns['Cluster'].unique():
#     cluster_name = f'Cluster {cluster}'
#     G.add_edge(cluster_name, 'total costs')

# Draw the graph
pos = nx.spring_layout(G)
fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

# Draw nodes with uniform color
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)


# Create lists to store edge colors and widths
edge_colors = []
edge_widths = []

# Determine color and width for each edge
for edge in G.edges():


    value = float(edge_labels[edge])  # Convert string to float
    if value < 0:
        print("red edge: ", edge)
        edge_colors.append('red')
    else:
        print("blue edge: ", edge)
        edge_colors.append('blue')
    
    # Optional: Adjust edge width based on absolute value
    edge_widths.append(0.1 + abs(value) * 2)  # Multiply by 2 for visibility, adjust as needed

# Draw edges with specified styles
nx.draw_networkx_edges(G, pos, 
                       edge_color=edge_colors, 
                       width=edge_widths, 
                       arrows=True, 
                       arrowsize=20, 
                       ax=ax)

# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)


plt.title('Clusters with Features and Total Costs')

# Create a table of clusters and their associated features
clustered_columns_sorted = clustered_columns.sort_values('Cluster')

table_data = clustered_columns_sorted.groupby('Cluster')['Feature'].apply(lambda x: '\n'.join(x)).reset_index()

table_data.columns = ['Cluster', 'Features']

# Annotate the right side with cluster information
text_str = ""

for cluster in clusters_sorted:
    cluster_name = f'Cluster {cluster}'
    features = sorted(clustered_columns[clustered_columns['Cluster'] == cluster]['Feature'].tolist())
    #text_str += f"$\\bf{{{cluster_name}}}$:\n" + "\n".join([f"- {feature} ({feature_avg_dict[feature]:.2f})" for feature in features]) + "\n\n"
    text_str += f"$\\bf{{{cluster_name}}}$:\n" + "\n".join([
        f"- {feature} ({feature_avg_dict[feature]:.2f})" 
        for feature in features 
        #if abs(feature_avg_dict[feature]) > 0.2
    ]) + "\n\n"

# Identify features not associated with any cluster
all_features = set(df_scaled_data_filtered.columns)

clustered_features = set(clustered_columns['Feature'])
not_clustered_features = sorted(all_features - clustered_features)

# Add "Not Clustered" features to the text annotation
if not_clustered_features:
    text_str += f"$\\bf{{Not Clustered}}$:\n" + "\n".join([f"- {feature} ({feature_avg_dict[feature]:.2f})" for feature in not_clustered_features]) + "\n\n"

# Add the text annotation to the plot
plt.annotate(text_str, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10, ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))
plt.subplots_adjust(right=0.75)  # Adjust the right space to make room for the table

# save plot
#plt.savefig('clusters_with_features_and_total_costs.png', dpi=300)  # Save the plot as a PNG file
plt.savefig(os.path.join(figure_path, f"clusters_with_features_and_total_costs.png"), dpi=300)

plt.show()