import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
data_path = 'dataset.csv'
df = pd.read_csv(data_path)

# The last column is the target and the rest are features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardizing the features before applying PCA
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA())
])

# Fit the model with X and apply the dimensionality reduction on X
pipeline.fit(X)
X_pca = pipeline.transform(X)

# Determine the number of components for 95% variance
pca = pipeline.named_steps['pca']  # Extract the PCA step from the pipeline
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(explained_variance >= 0.95) + 1  # +1 since index starts at 0

print(f"Number of components to retain 95% variance: {n_components_95}")

# Reducing the dataset dimensions with n_components_95
pipeline.set_params(pca__n_components=n_components_95)
X_reduced = pipeline.fit_transform(X)

# Visualization of the PCA components
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

# Saving the reduced data into a new CSV file
np.savetxt("reduced_dataset.csv", X_reduced, delimiter=",")
