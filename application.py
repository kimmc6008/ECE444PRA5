from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

def load_model():
    """Load the trained machine learning model and vectorizer from disk."""
    global loaded_model, vectorizer

    model_path = 'basic_classifier.pkl'
    vectorizer_path = 'count_vectorizer.pkl'

    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return "Fake News Detection API is running."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting whether a piece of text is real or fake news.
    Expects a JSON request with a 'text' field.
    """
    load_model()
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text field provided"}), 400

    text = data['text']

    # Check if 'text' is a list or a single string
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    else:
        return jsonify({"error": "Invalid format for text field"}), 400

    predictions = loaded_model.predict(vectorizer.transform(texts))

    results = ["FAKE" if pred == 1 else "REAL" for pred in predictions]

    return jsonify({'predictions': results})

if __name__ == "__main__":
    app.run(debug=True)
