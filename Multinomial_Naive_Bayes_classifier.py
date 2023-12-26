# create the dataset of student responses 
# clean and preprocess the text data 
# convert text data into numerical features using TF-IDF technique
# train the model to classify the responses 
# use the model to classify new responses and evaluate its performance 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')

# create the dataset
data = pd.DataFrame({
    'Response': ['I think photosynthesis is a process',
                 'Photosynthesis produces oxygen',
                 'Mitochondria is the powerhouse of the cell',
                 'The earth revolves around the sun',
                 'A square has four sides',
                 'Triangles have three angles'],
    'Category': ['Partially Correct', 'Correct', 'Incorrect', 'Correct', 'Correct', 'Incorrect']
})

# clean and preprocess the text data 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove all non-word characters and lower case the text
    text = re.sub(r'\W', ' ', text.lower())
    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

data['Processed_Response'] = data['Response'].apply(preprocess_text)

# Feature Extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['Processed_Response'])
y = data['Category']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multinomial Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting and evaluating the model
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
