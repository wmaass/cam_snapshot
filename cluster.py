import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data
data = [
    [0.00, 0.18228876, -0.25379637, -0.30412888, 0.13636838, -0.11423833, 0.35691261, -0.41748739, -0.03459309, 0.15985288, -0.11046083, 0.08823933],
    [0.00, 0.00, 0.34893014, 0.00, 0.21585593, 0.21719865, -0.79223856, 0.00, 0.32806831, -0.06664821, 0.00, 0.50302056],
    [0.30467009, -0.35213326, 0.38960797, -0.01898389, -0.33454679, 0.02692485, 0.78267548, 0.00, 0.26889685, -1.00, 0.00, 0.08729123],
    [0.0286336, 0.00, -0.84496969, -0.94572426, -0.36505063, 0.08367047, -0.17375104, -0.854621, -0.11827245, -0.04721455, 0.00997679, 0.05556349]
]
data = np.array(data)

data_T = data.T

print(data_T)

column_names = [
    "APR Risk of Mortality", "APR MDC Code", "CCS Procedure Description", "APR Severity of Illness Code", 
    "APR DRG Code", "CCS Procedure Code", "CCS Diagnosis Code", "APR Severity of Illness Description", 
    "APR MDC Description", "APR DRG Description", "APR Medical Surgical Description", "CCS Diagnosis Description"
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
optimal_clusters = 5

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
