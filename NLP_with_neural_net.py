# Create dataset of student responses.
# Tokenize the text data and convert  it into sequences 
# Create a model with Neural Network layers for text classification 
# train the model on the dataset and evaluate its performance.

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# Text Preprocessing
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Response'])
sequences = tokenizer.texts_to_sequences(data['Response'])
padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['Category'])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Building the Neural Network
model = Sequential([
    Embedding(5000, 16, input_length=20),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
