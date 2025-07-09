import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Generating synthetic dataset
X, y = make_blobs(n_samples=300, centers=3, n_features=4, cluster_std=1.0, random_state=42)

# Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(X)

# Post-process to balance clusters
def balance_clusters(X, labels, n_clusters):
    n_samples = len(labels)
    target_size = n_samples // n_clusters
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    
    for cluster in range(n_clusters):
        while cluster_sizes[cluster] > target_size:
            indices = np.where(labels == cluster)[0]
            if len(indices) == 0:
                break
            idx_to_move = np.random.choice(indices)
            other_clusters = [i for i in range(n_clusters) if i != cluster]
            min_dist = float('inf')
            best_cluster = -1
            for other_cluster in other_clusters:
                if cluster_sizes[other_cluster] < target_size:
                    distance = np.linalg.norm(X[idx_to_move] - X[labels == other_cluster], axis=1).min()
                    if distance < min_dist:
                        min_dist = distance
                        best_cluster = other_cluster
            if best_cluster >= 0:
                labels[idx_to_move] = best_cluster
                cluster_sizes[cluster] -= 1
                cluster_sizes[best_cluster] += 1
                
    return labels

print(X)

# Using the balance_clusters function
balanced_labels = balance_clusters(X, labels, n_clusters=3)

# Visualize the clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Agglomerative Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=balanced_labels)
plt.title("Balanced Agglomerative Clustering")

plt.show()
