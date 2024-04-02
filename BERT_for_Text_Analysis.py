from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
text = "Multivariable Calculus and Differential Equations are very interesting. Math is my favorite subject."

# Encode text
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_tensor = torch.tensor([input_ids])

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get predicted class (0 or 1) and score
predicted_class = torch.argmax(predictions).numpy()
confidence_score = torch.max(predictions).numpy()

# Mapping for class labels
class_names = ['Negative', 'Positive']

print(f"Sentence: {text}")
print(f"Sentiment: {class_names[predicted_class]}")
print(f"Confidence Score: {confidence_score}")

# BERT-based sentiment analysis pipeline (simpler approach)
nlp_pipeline = pipeline("sentiment-analysis")
result = nlp_pipeline(text)
print(f"Sentence: {text}")
print(f"Sentiment: {result[0]['label']}, with confidence {result[0]['score']}")
