# A list of learning materials and their corresponding categories is created.
# The CountVectorizer from sklearn is used to convert the textual data into numerical data, as ML algorithms require numerical input.
# The dataset is split into training and testing sets to evaluate the model's performance.
# The MultinomialNB classifier, which is suitable for text classification, is initialized and trained on the training data.
# Predictions are made on the test set, and the accuracy is calculated.


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

materials_updated = [
    "Integral calculus and its applications",
    "Newton's laws of motion and their implications",
    "The periodic table and chemical bonding",
    "Statistical analysis and probability theory",
    "Quantum mechanics and the double-slit experiment",
    "Organic chemistry and the structure of hydrocarbons"
]

categories_updated = ['Math', 'Physics', 'Chemistry', 'Math', 'Physics', 'Chemistry']

# Convert the text data into numerical data
X_updated = vectorizer.fit_transform(materials_updated)

# Split the dataset into training and testing sets
X_train_updated, X_test_updated, y_train_updated, y_test_updated = train_test_split(
    X_updated, categories_updated, test_size=0.2, random_state=42)

# Initialize the classifier
classifier_updated = MultinomialNB()

# Train the classifier
classifier_updated.fit(X_train_updated, y_train_updated)

# Predict on the test set
predictions_updated = classifier_updated.predict(X_test_updated)

# Calculate the accuracy
accuracy_updated = accuracy_score(y_test_updated, predictions_updated)

accuracy_updated, predictions_updated
