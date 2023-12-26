# create the dataset with features like study hours, previous scores, and other factors, along with their corresponding performance scores over time 
# prepare the data for regression analysis 
# Gradient Boosting or Random Forest Regression to handle nonlinear relationships and interactions between features 
# train the model on the dataset and evaluate its performance.


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# create the dataset
np.random.seed(42)
n_students = 100
n_weeks = 20
student_ids = np.repeat(range(n_students), n_weeks)
week_numbers = np.tile(range(n_weeks), n_students)
study_hours = np.random.normal(10, 2, n_students * n_weeks)
previous_scores = np.random.normal(75, 10, n_students * n_weeks)
performance_scores = previous_scores + np.random.normal(5, 1, n_students * n_weeks) * np.log(study_hours)

data = pd.DataFrame({
    'StudentID': student_ids,
    'WeekNumber': week_numbers,
    'StudyHours': study_hours,
    'PreviousScores': previous_scores,
    'PerformanceScores': performance_scores
})

# prepare the data for regression analysis 

# Split the data into training and testing sets
X = data[['StudentID', 'WeekNumber', 'StudyHours', 'PreviousScores']]
y = data['PerformanceScores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Plotting
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Performance Prediction')
plt.show()

 
