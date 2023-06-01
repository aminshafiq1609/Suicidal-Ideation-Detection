import flask
from flask import Flask, request, jsonify
import joblib 
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

def clean_text(text):
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

    return tokens

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data sent to this endpoint
    df = pd.DataFrame(data, index=[0])  # Convert data to a pandas DataFrame
    
    # Preprocess the "Text" column
    df['Text'] = df['Text'].apply(clean_text)
    
    prediction = model.predict(df)  # Predict using your trained model
    return jsonify(prediction.tolist())  # Return the prediction as JSON

if __name__ == '__main__':
    model = joblib.load('C:\\Users\\amins\\Desktop\\Project\\random_forest_model.pkl')  # Load your trained model
    app.run(port=5000)  # Start the Flask app


