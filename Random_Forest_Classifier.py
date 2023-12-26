# Create the dataset representing student performance across different topics 
# prepare features that influence the recommendation, performance scores, time spent on each topic, etc.
# use  Random Forest model to train on this data 
# use the trained model to predict the next best topic for a student 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate the  dataset
np.random.seed(42)
n_students = 100
topics = ['Algebra', 'Geometry', 'Trigonometry', 'Calculus', 'Statistics']
n_topics = len(topics)

# Student performance data
data = pd.DataFrame({
    'StudentID': np.repeat(range(n_students), n_topics),
    'Topic': np.tile(topics, n_students),
    'Score': np.random.randint(50, 100, n_students * n_topics),
    'TimeSpent': np.random.randint(30, 120, n_students * n_topics)
})

# Identifying the target variable: Next Best Topic  
data['NextBestTopic'] = data['Topic'].shift(-1).fillna('Algebra')

# convert  categorical data to numerical
data = pd.get_dummies(data, columns=['Topic'])

# Splitting the dataset
X = data.drop(columns=['StudentID', 'NextBestTopic'])
y = data['NextBestTopic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting the next best topic
predictions = rf_classifier.predict(X_test)

# Calculate accuracy (just for demonstration purposes)
accuracy = accuracy_score(y_test, predictions)


