# SVM classifier with a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer 
# since this approach is often more effective for text classification tasks since it considers the overall importance of each word in the dataset 
# Use TfidfVectorizer instead of CountVectorizer for better feature extraction 
# Use Support Vector Classifier sicne it is often more effective for text classification 
# TF-IDF approach helps to understand the context of words more effectively  and SVMs are generally powerful for classification tasks 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

 
materials = [
    "Integral calculus and its applications",
    "Newton's laws of motion and their implications",
    "The periodic table and chemical bonding",
    "Statistical analysis and probability theory",
    "Quantum mechanics and the double-slit experiment",
    "Organic chemistry and the structure of hydrocarbons"
]

categories = ['Math', 'Physics', 'Chemistry', 'Math', 'Physics', 'Chemistry']

# Using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Convert the text data into numerical data
X = tfidf_vectorizer.fit_transform(materials)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = svm_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

 
