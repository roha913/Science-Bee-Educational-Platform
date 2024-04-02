import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

# Load datasets
iris = load_iris()
boston = load_boston()

# For Classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# For Regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    boston.data, boston.target, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
param_grid_c = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_c = GridSearchCV(estimator=dt_classifier, param_grid=param_grid_c, cv=5, n_jobs=-1, verbose=2)
grid_search_c.fit(X_train_c, y_train_c)
best_classifier = grid_search_c.best_estimator_

# Evaluate classifier
y_pred_c = best_classifier.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"Classification Accuracy: {accuracy:.4f}")
print(f"Best Classifier Parameters: {grid_search_c.best_params_}")

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
param_grid_r = {
    'criterion': ['mse', 'friedman_mse', 'mae'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_r = GridSearchCV(estimator=dt_regressor, param_grid=param_grid_r, cv=5, n_jobs=-1, verbose=2)
grid_search_r.fit(X_train_r, y_train_r)
best_regressor = grid_search_r.best_estimator_

# Evaluate regressor
y_pred_r = best_regressor.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
print(f"Regression MSE: {mse:.4f}")
print(f"Best Regressor Parameters: {grid_search_r.best_params_}")

# Visualize the tree
# Classifier
plt.figure(figsize=(20,10))
tree.plot_tree(best_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Best Decision Tree Classifier")
plt.show()

# Regressor
plt.figure(figsize=(20,10))
tree.plot_tree(best_regressor, filled=True, feature_names=boston.feature_names)
plt.title("Best Decision Tree Regressor")
plt.show()
