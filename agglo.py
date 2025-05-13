from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
# Assuming you have already loaded and preprocessed the data as 'X_scaled'
# Apply Agglomerative Clustering
n_clusters = 3 # Specify number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(X_scaled)
# Evaluation metrics (assuming ground truth labels are available in
'y_true')
y_true = [0, 1, 2] # Replace this with your actual ground truth labels (if
available)
# Silhouette Score
silhouette = silhouette_score(X_scaled, agg_labels)
print(f"Agglomerative Silhouette Score: {silhouette:.4f}")
# Adjusted Rand Index
ari = adjusted_rand_score(y_true, agg_labels)
print(f"Agglomerative Adjusted Rand Index: {ari:.4f}")