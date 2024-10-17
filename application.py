from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Initialize Flask application
app = Flask(__name__)

# Load the pickled model and vectorizer
def load_model():
    """Load the trained machine learning model and vectorizer from disk."""
    global loaded_model, vectorizer  # Declare global variables

    model_path = 'basic_classifier.pkl'
    vectorizer_path = 'count_vectorizer.pkl'

    # Load model
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Load vectorizer
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

# Route to check if the app is running
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
        texts = [text]  # Convert to a list for consistent processing
    elif isinstance(text, list):
        texts = text  # Process as a list of strings
    else:
        return jsonify({"error": "Invalid format for text field"}), 400

    # Vectorize and predict
    predictions = loaded_model.predict(vectorizer.transform(texts))

    # Interpret predictions
    results = ["FAKE" if pred == 1 else "REAL" for pred in predictions]

    # Return prediction result as JSON
    return jsonify({'predictions': results})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
