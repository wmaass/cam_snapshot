import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg

import sys

# Function to count non-empty bins (colors) in each column
def count_non_empty_bins_per_column(column, bins):
    # Bin the data in the column
    # binned_data = pd.cut(column.values, bins=bins, include_lowest=True)
    # Count unique non-empty bins
    # non_empty_bins = binned_data.value_counts().count()

    # Use pd.cut to bin the data in the column
    binned_data = pd.cut(column, bins=bins, include_lowest=True)

    # Directly count non-null entries in binned_data, which represent non-empty bins
    non_empty_bins = len(np.unique(binned_data[~pd.isnull(binned_data)]))

    #print("non_empty_bins: ", non_empty_bins)

    # print(non_empty_bins)
    return non_empty_bins

def create_custom_heatmap(df, bins, custom_cmap, vmin=1, vmax=4, cluster_num=2):
    """
    Creates a heatmap with customizations including vertical lines on changes in the number of unique colors used in each column.

    Parameters:
    - df: DataFrame to visualize.
    - bins: List of bin edges for categorizing the data.
    - custom_cmap: Custom colormap for the heatmap.
    - vmin: Minimum value for colormap scaling.
    - vmax: Maximum value for colormap scaling.
    - cluster_num: The index of the column for the first heatmap, if not provided defaults to 2.
    """
    # Assume count_non_empty_bins_per_column is defined elsewhere
    
    # Calculate the number of non-empty bins (colors) for each column
    color_counts_per_column = [count_non_empty_bins_per_column(df[col], bins) for col in df]

    # Create the heatmap for the n-th column
    fig1, ax1 = plt.subplots(figsize=(5, 6))  # Adjust the figure size as needed
    fig1.subplots_adjust(left=0.5, top=0.95, bottom=0.1)  # Adjusting the left margin for p1

    p1 = sns.heatmap(df.iloc[:, [cluster_num]], cmap=custom_cmap, xticklabels=False, yticklabels=True, vmin=vmin, vmax=vmax, annot=False, linewidths=0.5, linecolor='black')
    # Set the x-axis label to the value of 'cluster_num'
    ax1.set_xlabel(f'Cluster Number: {cluster_num+1}', fontsize=12)

    # Creating the main heatmap with adjusted left margin
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.3, top=0.95, bottom=0.2)

    # Create the main heatmap
    p = sns.heatmap(df, cmap=custom_cmap, xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, annot=False, linewidths=0.5, linecolor='black')

    # Set x-tick labels to the number of unique colors used in each column
    plt.xticks(ticks=np.arange(len(df.columns)) + 0.5, labels=color_counts_per_column, rotation=45, ha='right')

    plt.yticks(rotation=0, fontsize=8)
    plt.title('Cluster Values for Each Feature')
    plt.xlabel('Cluster Bin Counts')
    plt.ylabel('Feature Name')

    # Add thick vertical lines where x-labels increase
    for i in range(1, len(color_counts_per_column)):
         if color_counts_per_column[i] != color_counts_per_column[i - 1]:
             plt.axvline(x=i, color='black', linewidth=2)

    return(p1, p)

def d_plot(df, cluster_num):

    # Preparing data for 3D surface plot
    x = np.arange(df.shape[1])  # Cluster numbers as x-axis
    y = np.arange(df.shape[0])  # Facilities as y-axis
    x, y = np.meshgrid(x, y)  # Create a mesh grid for x and y
    z = df.to_numpy()  # Values as z-axis

    vmin, vmax = 0, 49

    # Define 10 custom colors
    colors = [
        "#FF0000", "#FF7F00", "#FFFF00", "#7FFF00", "#00FF00" , "#00FF7F", "#00FFFF", "#007FFF", "#0000FF", "#7F00FF"
    ]
    custom_cmap = ListedColormap(colors)

    # Calculate the value range for each color
    bins = np.linspace(vmin, vmax, len(colors) + 1)

    p1, p = create_custom_heatmap(df, bins, custom_cmap, cluster_num=cluster_num)

    return(p1, p)


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python script.py <file1.csv>")
    #     sys.exit(1)

    #file1= sys.argv[1]
    #file2= sys.argv[2]
    #cluster_num = int(sys.argv[3])    

    file1 = "HIS17"
#    file2 = "HIS17-2"
    file2 = "HIS17-two-classes"
    cluster_num = 4

    # file1 = "LOAN"
    # file2 = "LOAN-2"
    # cluster_num = 1

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