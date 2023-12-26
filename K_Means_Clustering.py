# create the dataset with features like scores in Math, Physics, and Chemistry 
# standardize the data for clustering 
# use the Elbow Method to determine the optimal number of clusters 
# Use K-Means Clustering to cluster the students based on their performance 
# analyze the characteristics of each cluster.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# create the dataset
np.random.seed(42)
n_students = 200
math_scores = np.random.normal(70, 10, n_students)
physics_scores = np.random.normal(65, 15, n_students)
chemistry_scores = np.random.normal(75, 8, n_students)

data = pd.DataFrame({
    'Math': math_scores,
    'Physics': physics_scores,
    'Chemistry': chemistry_scores
})

# standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choosing the number of clusters with the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Adding the cluster data to the original dataset
data['Cluster'] = clusters

# Analysis of Clusters
# Displaying average scores per cluster
average_scores_per_cluster = data.groupby('Cluster').mean()
 
