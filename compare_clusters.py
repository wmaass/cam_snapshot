import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
import matplotlib.lines as mlines
import sys
from tabulate import tabulate

import concept_def_oo as cdo
from concept_def_oo import * # that's the problem for reloading issues!!! use calign21.function instead
import plot3d as p3d
from  plot3d import * # that's the problem for reloading issues!!! use calign21.function instead

import itertools

def prepare_maxdiff_data(metrics_df, metric_name, offset=0):
    """
    Prepare data for plotting the metric values and finding the knee point, excluding clusters smaller than 'offset'.

    Parameters:
    - metrics_df: DataFrame containing the metrics.
    - metric_name: The name of the metric column to analyze ('ARI', 'NMI', or 'FMI').
    - offset: The minimum size (or other criteria) to include a cluster in the analysis.

    Returns:
    - A dictionary containing x, y, mdiff_max, and the y-value at mdiff_max for plotting.
    """
    metrics_df_copy = metrics_df.copy()
    metrics_df_copy = metrics_df_copy.drop(metrics_df_copy.index[:offset])
    x = metrics_df_copy.index
    y = metrics_df_copy[metric_name].values
    mdiff = metrics_df_copy[metric_name].diff()
    mdiff_max = mdiff.idxmax()

    return {
        'x': x + offset,
        'y': y,
        'mdiff_max': mdiff_max,
        'y_mdiff_max': metrics_df_copy[metric_name][mdiff_max]
    }

def compare_cluster_assignments(file1, file2):
    # Read file1 and file2 into dataframes df1 and df2. Assume the first column is the index.
    clusters1 = pd.read_csv("./clusters/column_clusters_" + file1 + ".csv", index_col=0)
    clusters2 = pd.read_csv("./clusters/column_clusters_" + file2 + ".csv", index_col=0)
    
    # drop rows nan_rows in cluster 1 and cluster2
    clusters1 = clusters1.dropna()
    clusters2 = clusters2.dropna()
    
    # collect all index names with rows that do not contain NaN values
    c1 = clusters1.index
    c2 = clusters2.index
    
    # drop column 'Cluster_1' if it exists
    if 'Cluster_1' in clusters1.columns:
        clusters1 = clusters1.drop('Cluster_1', axis=1)
        clusters1 = clusters1.drop('Cluster_2', axis=1)

    if 'Cluster_1' in clusters2.columns:
        clusters2 = clusters2.drop('Cluster_1', axis=1)
        clusters2 = clusters2.drop('Cluster_2', axis=1)

    cluster_offset = 3

    # set flag if list c1 is subset of list c2 or list c2 is subset of list c1
    is_subset1 = set(c1).issubset(c2)
    is_subset2 = set(c2).issubset(c1)

    # if is_subset1 or is_subset2 is True, determine percentage of subset
    if is_subset1:
        #print(file1 + " is a subset of " + file2)
        # assign subset of c1 to c2 to list subset1
        subset1 = set(c1).intersection(c2)
        clusters2 = clusters2.loc[list(subset1)]
        perc1 = len(c1) / len(c2) * 100
        print(f'subset1: {subset1} with {perc1}%' )

    elif is_subset2:
        print(file2 + " is a subset of " + file1)
        # assign subset of c1 to c2 to list subset1
        subset2 = set(c2).intersection(c1)
        clusters1 = clusters1.loc[list(subset2)]
        perc2 = len(c2) / len(c1) * 100
        print(f'subset2: {subset2} with {perc2}%' )
    else:
        print("No subset found")

    # save clusters1 and clusters2 to csv files
    clusters1.to_csv("./clusters/cluster1_intersect.csv")
    clusters2.to_csv("./clusters/cluster2_intersect.csv")

    # Ensure headers are read correctly
    clusters1.columns = clusters1.columns.astype(str)
    clusters2.columns = clusters2.columns.astype(str)

    #print(f"Clusters1: {clusters1.columns}")
    #print(f"Clusters2: {clusters2.columns}")

    # Get the order of features (index) from clusters1
    feat_order = clusters1.index

    # Reorder clusters2 to match the feature order of clusters1
    clusters2_ordered = clusters2.reindex(feat_order)

    # replace NaN values with 0
    clusters2_ordered = clusters2_ordered.fillna(0)
    #print(f"Clusters2 ordered: {clusters2_ordered.columns}")

    # Initialize a DataFrame to store the metrics
    metrics_df = pd.DataFrame(columns=['Column', 'ARI', 'NMI', 'FMI'])

    # tabular print of clusters1 and clusters2_ordered

    #print(tabulate(clusters1, headers='keys', tablefmt='grid'))
    #print(tabulate(clusters2_ordered, headers='keys', tablefmt='grid'))

    i = 0
    # Iterate over all columns and compare cluster assignments
    for col in clusters1.columns:
        #print("Cluster column i: ", i)
        i += 1

        if col in clusters2_ordered.columns:
            # Calculate ARI, NMI, and FMI
            ari_score = adjusted_rand_score(clusters1[col], clusters2_ordered[col])
            nmi_score = normalized_mutual_info_score(clusters1[col], clusters2_ordered[col])
            fmi_score = fowlkes_mallows_score(clusters1[col], clusters2_ordered[col])

            # create dataframe with ari_score, nmi_score, fmi_score
            df_new = pd.DataFrame({'Column': [col], 'model A': [file1], 'model B': [file2], 'ARI': [ari_score], 'NMI': [nmi_score], 'FMI': [fmi_score]})

            # if file fn does not exist, create it
            # and add df_new to it
            if not os.path.isfile("./clusters/similarity.csv"):
                df_new.to_csv("./clusters/similarity.csv", index=False)
            else:
                # append df_new to file
                with open("./clusters/similarity.csv", 'a') as f:
                    df_new.to_csv(f, header=False, index=False)

            # concat df_new to metrics_df
            metrics_df = pd.concat([metrics_df, df_new], ignore_index=True)

            # # Append the results to the metrics DataFrame
            # metrics_df = metrics_df.concat({'Column': col, 'ARI': ari_score, 'NMI': nmi_score, 'FMI': fmi_score}, ignore_index=True)
        else:
            print(f"Column {col} not found in both DataFrames.")


    # Save the metrics DataFrame to a CSV file
    output_file = "./clusters/cluster_analytics.csv"
    metrics_df.to_csv(output_file, index=False)
    #print(f"Metrics saved to {output_file}")
    
    #ari_max = find_and_plot_maxdiff(metrics_df, 'ARI')
    #nmi_max = find_and_plot_maxdiff(metrics_df, 'NMI')
    #fmi_max = find_and_plot_maxdiff(metrics_df, 'FMI')

    # Assuming metrics_df is your DataFrame and is already defined
    metrics = ['ARI', 'NMI', 'FMI']
    metrics_full = ['Adjusted Rand Index (ARI)', 'Normalized Mutual Information (NMI)', 'Fowlkes-Mallows score (FMI)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    line_styles = ['dashdot', 'dotted', ':']  # Example line styles
    offsets = [-0.1, 0, 0.1]  # Small offsets to apply to x-position of vertical lines
    alpha = 0.7  # Transparency

    offset = 0  # Adjust as needed
    x_offset = 0  # Adjust as needed

    plt.figure(figsize=(10, 6))

    # Create a custom legend handle for the red circled item
    red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, markerfacecolor='none', markeredgecolor='red', label='Cluster with the highest gradient')


    for metric_fn, metric_name, color, line_style, plot_offset in zip(metrics_full, metrics, colors, line_styles, offsets):
        data = prepare_maxdiff_data(metrics_df, metric_name, offset=0)
        plt.plot(data['x'], data['y'], marker='o', linestyle=line_style, label=metric_fn, color=color)
        
        # Apply offset for plotting, not for x-axis labeling
        adjusted_x = data['mdiff_max'] + plot_offset
        plt.text(adjusted_x, data['y_mdiff_max'], f'   {data["y_mdiff_max"].round(3)}', fontsize=12, color=color, ha='right', va='bottom')
        plt.plot([adjusted_x, adjusted_x], [data['y_mdiff_max'], 0], linestyle='--', color=color, alpha=alpha)
        plt.scatter(data['mdiff_max'], data['y_mdiff_max'], s=100, facecolors='none', edgecolors='red', linewidths=2)

    # Adjust x-axis labels to include the offset
    adjusted_x_labels = [x + x_offset for x in data['x']]
    plt.xticks(ticks=data['x'], labels=adjusted_x_labels, fontsize=8)

    # change x-axis labels
    plt.xticks(data['x'], data['x']+cluster_offset, fontsize=8) 

    plt.xlabel('Adjusted Number of Clusters')
    plt.ylabel('Metric Value')
    plt.title('Cluster Analytics: Benchmark ' + file1 + ' (Gold Standard) against ' + file2)

    # Add the custom legend handle to the existing legend handles
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(red_circle)  # Add the custom legend handle for the red circle

    plt.legend(handles=handles, loc='best')

    plt.tight_layout()

    # save the plot to a file
    plt.savefig("./clusters/cluster_analytics.png")
    plt.show()

    m = metrics_df.iloc[:, 1:4]

    print(m)

    # determine mdiff that contains the difference of each consecutive element in a column of m
    mdiff = m.diff()

    return(clusters1, clusters2_ordered)


def create_co_association_matrix(df):
    n_features = df.shape[0]
    co_matrix = np.zeros((n_features, n_features))
    
    for _, row in df.items():
        for i in range(n_features):
            for j in range(n_features):
                if row.iloc[i] == row.iloc[j]:
                    co_matrix[i, j] += 1

    co_matrix /= df.shape[1]  # Normalize by the number of time points
    return co_matrix

         
    

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <file1.csv> <file2.csv>")
    #     sys.exit(1)
    
    names = ["HIS17-random", "HIS17-Benchmark"] # ["HIS17-2"] #, "HIS17-two-classes"] # ['LOAN', 'LOAN-2'] # ["HIS17-2", "HIS17-two-classes"] # "HC-four-classes"] # , "HIS17-onehot-random", "HIS17-onehot"]
    #names = ['LOAN', 'LOAN-2'] # ["HIS17-2", "HIS17-two-classes"] # "HC-four-classes"] # , "HIS17-onehot-random", "HIS17-onehot"]

    benchmark = "HIS17-Benchmark"
    #benchmark = "LOAN"

    cluster_num = 3

    # delete files in clusters directory
    import os
    c_folder = './clusters/'
    fn = "similarity.csv"

    # delete fn in folder
    if os.path.exists(c_folder + fn):
        os.remove(c_folder + fn)
    
    for app_name in names:
        analysis = FeatureAnalysis(app_name, benchmark)
        analysis.extract_and_save_concepts()
        analysis.init_cooccurence_matrix()
        analysis.update_cooccurence_matrix()
        analysis.plot_heatmap(clustered=False)  # Setzen Sie clustered auf True, um die nicht-geclusterte Heatmap zu plotten
        analysis.plot_heatmap(clustered=True)  # Setzen Sie clustered auf True, um die geclusterte Heatmap zu plotten

    # compare cluster assignments
        
    # Generate and iterate over all binary combinations
    for combination in itertools.combinations(names, 2):

        #print(combination)

        file1, file2 = combination

        print(f"Comparing {file1} and {file2}")

        #file1, file2 = sys.argv[1], sys.argv[2]
        c1, c2 = compare_cluster_assignments(file1, file2)

        #print(tabulate(c1, headers='keys', tablefmt='grid'))
        #print(tabulate(c2, headers='keys', tablefmt='grid'))

        co_matrix_c1 = create_co_association_matrix(c1)
        #print(tabulate(co_matrix_c1, headers='keys', tablefmt='grid'))

        co_matrix_c2 = create_co_association_matrix(c2)
        #print(tabulate(co_matrix_c2, headers='keys', tablefmt='grid'))

        # Calculate the Frobenius norm of the difference between the two matrices
        matrix_difference = co_matrix_c1 - co_matrix_c2
        frobenius_norm = np.linalg.norm(matrix_difference, 'fro')

        print(f"Frobenius Norm of the difference: {frobenius_norm}")

        # calculate the cosine similarity between the two matrices
        dot_product = np.dot(co_matrix_c1.flatten(), co_matrix_c2.flatten())
        norm_c1 = np.linalg.norm(co_matrix_c1)
        norm_c2 = np.linalg.norm(co_matrix_c2)
        cosine_similarity = dot_product / (norm_c1 * norm_c2)

        print(f"Cosine Similarity between {file1} and {file2}: {cosine_similarity}")

        # plot clustering results as heatmaps: plot3d.py

        # Read file1 and file2 into dataframes df1 and df2. Assume the first column is the index.
        df1 = pd.read_csv("./clusters/column_clusters_" + file1 + ".csv", index_col=0)
        df2 = pd.read_csv("./clusters/column_clusters_" + file2 + ".csv", index_col=0)

        # order df2 according to the index of df1
        df2 = df2.reindex(df1.index)

        dfs = [df1, df2]

        images = []
        filenames1 = []
        filenames = []
        for i, df in enumerate(dfs):
            p1, p = d_plot(df, cluster_num)  # Create the heatmap

            filename1 = f'./clusters/heatmap_3col_{i}.png'
            p1.get_figure().savefig(filename1)
            filenames1.append(filename1)
            plt.close(p1.get_figure())  # Close the figure to free memory

            filename = f'./clusters/heatmap_{i}.png'
            p.get_figure().savefig(filename)
            filenames.append(filename)
            plt.close(p.get_figure())  # Close the figure to free memory

        # Now, load and show all plots in images in one figure
        fig, axes = plt.subplots(1, len(dfs), figsize=(10, 5))  # Adjust the number of axes based on the number of dfs
        for i, filename in enumerate(filenames1):
            img = mpimg.imread(filename)
            axes[i].imshow(img)
            axes[i].axis('off')  # Hide axes

        # Now, load and show all plots in images in one figure
        fig, axes = plt.subplots(1, len(dfs), figsize=(10, 5))  # Adjust the number of axes based on the number of dfs
        for i, filename in enumerate(filenames):
            img = mpimg.imread(filename)
            axes[i].imshow(img)
            axes[i].axis('off')  # Hide axes

        plt.tight_layout()

        # save the figure
        plt.savefig('./clusters/heatmaps.png')

        plt.show()
