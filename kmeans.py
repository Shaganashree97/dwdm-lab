from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score, adjusted_rand_score 
from sklearn.preprocessing import StandardScaler 
# Assuming you have already loaded the data as 'data' 
# Preprocess the data (standardize it) 
X = data.values  # Assuming data is numeric 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Apply K-Means clustering 
n_clusters = 3  # Specify number of clusters 
kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
kmeans_labels = kmeans.fit_predict(X_scaled) 
# Evaluation metrics (assuming ground truth labels are available in 'y_true') 
y_true = [0, 1, 2]  # Replace this with your actual ground truth labels (if available) 
# Silhouette Score 
silhouette = silhouette_score(X_scaled, kmeans_labels) 
print(f"K-Means Silhouette Score: {silhouette:.4f}") 
# Adjusted Rand Index 
ari = adjusted_rand_score(y_true, kmeans_labels) 
print(f"K-Means Adjusted Rand Index: {ari:.4f}") 