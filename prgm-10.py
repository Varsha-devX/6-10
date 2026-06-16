prgm-10
code:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
# 1. Simulate customer RFM data 
np.random.seed(42) 
n_customers = 200 
data = pd.DataFrame({ 
'CustomerID': range(1, n_customers + 1), 
'Recency': np.random.randint(1, 100, size=n_customers),     
# days since last purchase 
'Frequency': np.random.randint(1, 20, size=n_customers),    # number of purchases 
'Monetary': np.random.randint(100, 10000, size=n_customers) # total money spent 
}) 
# 2. Standardize the RFM features 
scaler = StandardScaler() 
rfm_scaled = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary']]) 
# 3. Find optimal number of clusters using the Elbow Method 
sse = [] 
for k in range(1, 11): 
kmeans = KMeans(n_clusters=k, random_state=42) 
kmeans.fit(rfm_scaled) 
sse.append(kmeans.inertia_) 
# Plot Elbow Curve 
plt.figure(figsize=(6, 4)) 
plt.plot(range(1, 11), sse, marker='o') 
plt.title("Elbow Method for Optimal K") 
plt.xlabel("Number of clusters") 
plt.ylabel("SSE (Inertia)") 
plt.grid(True) 
plt.show() 
# 4. Apply K-Means with optimal clusters (e.g., k=4) 
kmeans = KMeans(n_clusters=4, random_state=42) 
data['Cluster'] = kmeans.fit_predict(rfm_scaled) 
# 5. Visualize Clusters 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(data=data, x='Recency', y='Monetary', hue='Cluster', palette='Set2') 
plt.title("Customer Segmentation (Recency vs Monetary)") 
plt.show() 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(data=data, x='Frequency', y='Monetary', hue='Cluster', palette='Set1') 
plt.title("Customer Segmentation (Frequency vs Monetary)") 
plt.show()