import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import xgboost as xgb 
import matplotlib.pyplot as plt 

# Load your dataset using pandas (Replace 'your_dataset.csv' with the actual file name) 
# If you're using a different format, pandas supports .csv, .xlsx, .json, etc. 
# For example, loading CSV: 
df = pd.read_csv('your_dataset.csv') 
# Check the first few rows of the dataset 
print(df.head()) 
# Preprocessing:  
# Assume that the target variable is 'target' and the rest are features 
X = df.drop('target', axis=1)  # Features (input variables) 
y = df['target']               # Target (output variable) 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42) 
# Scale features (important for SVM, KNN, etc.) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

# 1. Logistic Regression 
print("\nLogistic Regression:") 
log_reg = LogisticRegression(max_iter=200) 
log_reg.fit(X_train, y_train) 
y_pred_log_reg = log_reg.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}") 
print(confusion_matrix(y_test, y_pred_log_reg)) 
print(classification_report(y_test, y_pred_log_reg)) 

# 2. Random Forest 
print("\nRandom Forest:") 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}") 
print(confusion_matrix(y_test, y_pred_rf)) 
print(classification_report(y_test, y_pred_rf)) 

# 3. Support Vector Machine (SVM) 
print("\nSupport Vector Machine (SVM):") 
svm = SVC(kernel='linear', random_state=42) 
svm.fit(X_train, y_train) 
y_pred_svm = svm.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}") 
print(confusion_matrix(y_test, y_pred_svm)) 
print(classification_report(y_test, y_pred_svm)) 

# 4. K-Nearest Neighbors (KNN) 
print("\nK-Nearest Neighbors (KNN):") 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 
y_pred_knn = knn.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn)}") 
print(confusion_matrix(y_test, y_pred_knn)) 
print(classification_report(y_test, y_pred_knn)) 

# 5. Decision Trees 
print("\nDecision Trees:") 
dt = DecisionTreeClassifier(random_state=42) 
dt.fit(X_train, y_train) 
y_pred_dt = dt.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt)}") 
print(confusion_matrix(y_test, y_pred_dt)) 
print(classification_report(y_test, y_pred_dt)) 

# 6. XGBoost 
print("\nXGBoost:") 
xg_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') 
xg_clf.fit(X_train, y_train) 
y_pred_xgb = xg_clf.predict(X_test) 
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}") 
print(confusion_matrix(y_test, y_pred_xgb)) 
print(classification_report(y_test, y_pred_xgb)) 

# Optional: Visualize Decision Tree (if it's a small dataset) 
plt.figure(figsize=(12, 8)) 
from sklearn.tree import plot_tree 
plot_tree(dt, filled=True, feature_names=X.columns, class_names=[str(i) for 
i in np.unique(y)], rounded=True) 
plt.title("Decision Tree Visualization") 
plt.show() 