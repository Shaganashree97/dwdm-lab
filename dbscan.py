import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generate a synthetic dataset with 2 features (for visualization)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Standardize the data (important for DBSCAN)
X = StandardScaler().fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)  # eps is the radius, min_samples is the minimum points per cluster
labels = dbscan.fit_predict(X)

# Number of clusters found (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Estimated Number of Clusters: {n_clusters}")

# Evaluation Metrics (only if there are clusters found)
if n_clusters > 1:
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
else:
    print("Silhouette Score not computed (Only one cluster found).")

# Plot the DBSCAN Clustering Results
plt.figure(figsize=(6, 5))
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'yellow']

for label, color in zip(unique_labels, colors):
    if label == -1:  # Noise points are black
        color = 'black'
    plt.scatter(X[labels == label, 0], X[labels == label, 1], c=color, label=f'Cluster {label}', edgecolors='k')

plt.title("DBSCAN Clustering Output")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()