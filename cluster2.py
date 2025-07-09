import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# print(data)

# data = [[ 0.        ,  1.        ,  0.4567613 ,  0.        ,  0.        ,
#          0.        ,  0.01381145,  0.        ,  0.57926009],
#        [-1.        ,  0.74845414, -0.0157683 ,  0.        , -0.13388085,
#          0.        ,  0.12330964,  0.        , -0.2440346 ],
#        [ 0.        , -0.05408326, -0.3167112 ,  0.        ,  0.        ,
#         -0.07654884,  0.11234016, -0.0024075 ,  1.        ],
#        [-0.00013387, -0.36714584, -0.49365369,  0.06330805,  0.08877024,
#          0.        ,  0.        ,  0.        ,  0.        ]]

data = [[0.0, -0.2997097622806384, 0.0, -0.0502845480041499], 
        [0.0, -0.0464091181488265, 0.0887918621640039, 0.4693522698412302], 
        [-0.0137850223729923, -0.1897969140040097, -0.2748224804816354, 0.020167191240287], 
        [0.1712283070812435, -0.1755547355751572, -0.0184974850903262, -0.0689396573160443], 
        [0.3634701954336576, -0.3674300138330707, 0.0010124818602175, 0.0107138413693882]]

data = np.array(data)

print(data)


data_T = data #.T
print(data_T)
column_names = ['Hospital Service Area', 'Hospital County', 'Facility Name', 'Permanent Facility Id', 'Operating Certificate Number'
]

# Determining the optimal number of clusters for columns
wcss = []
max_clusters = data_T.shape[0]
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_T)
    print(f"Cluster {i}: {kmeans.inertia_}")
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow method, decide the optimal number of clusters
optimal_clusters = 2

# Fitting KMeans with the optimal number of clusters to the transposed data
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(data_T)

# Reducing dimensions using PCA for 2D visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_T)

# Visualizing the clusters of columns in 2D
plt.figure(figsize=(10, 7))

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'grey', 'black', 'orange', 'purple', 'brown']

for i in range(optimal_clusters):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1], label=f"Cluster {i}", c=colors[i])
    for j, txt in enumerate(column_names):
        if clusters[j] == i:  # If the feature is in the current cluster
            plt.annotate(txt, (data_2d[j, 0], data_2d[j, 1]), size=10, color=colors[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='*', label='Centroids')
plt.title('Clusters of Features (Columns) in 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()
