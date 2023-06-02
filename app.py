import flask
from flask import Flask, request, jsonify
import joblib 
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained model
model = joblib.load('C:\\Users\\amins\\Desktop\\Project\\random_forest_model.pkl')

# Initialize a SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize a StandardScaler
scaler = StandardScaler()

def clean_text(text):
    # Your existing clean_text function here
    # Convert to lowercase
    text = text.lower()

    # Remove non-letter characters and punctuation
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Perform stemming on the tokens
    porter = nltk.PorterStemmer()
    tokens = [porter.stem(token) for token in tokens]

    # Convert tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def preprocess_text(text):
    # Apply the clean_text function
    cleaned_text = clean_text(text)

    # Extract emotion features
    sentiment_scores = sia.polarity_scores(cleaned_text)
    compound = sentiment_scores['compound']

    # Extract statistical features
    text_length = len(cleaned_text)
    average_word_length = np.mean([len(word) for word in cleaned_text.split()]) if cleaned_text else 0

    # Create a DataFrame with the features
    feature_df = pd.DataFrame({
        'compound': [compound],
        'text_length': [text_length],
        'average_word_length': [average_word_length]
    })

    # Standardize the features
    X = scaler.transform(feature_df)

    return X

@app.route('/predict', methods=['POST'])

def predict():
    data = request.json  # Get the JSON data sent to this endpoint
    text = data['Text']  # Extract the text from the data

    # Preprocess the text
    X = preprocess_text(text)
    
    # Predict using your trained model
    prediction = model.predict(X)
    
    return jsonify(prediction.tolist())  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(port=5000)  # Start the Flask app



