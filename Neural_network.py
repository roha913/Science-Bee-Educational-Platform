# create the dataset with student features and performance scores 
# prepare the data for the neural network 
# build the multi-layer neural network suitable for regression 
# train the network with the dataset 
# test the model's performance on unseen data 


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# create the dataset
np.random.seed(42)
n_students = 1000
study_hours = np.random.normal(5, 2, n_students)
attendance = np.random.choice([0, 1], n_students, p=[0.2, 0.8])
previous_scores = np.random.normal(70, 10, n_students)
performance_scores = previous_scores + study_hours * 5 + attendance * 10

data = pd.DataFrame({
    'StudyHours': study_hours,
    'Attendance': attendance,
    'PreviousScores': previous_scores,
    'PerformanceScores': performance_scores
})

# prepare the data for the neural network
X = data[['StudyHours', 'Attendance', 'PreviousScores']]
y = data['PerformanceScores']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building the Neural Network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
