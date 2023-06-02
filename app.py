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

    # Convert tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def preprocess_data(df):
    # Extract emotion features using VADER sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['Text'].apply(lambda x: sia.polarity_scores(' '.join(x)))
    df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

    # Extract statistical features
    df['text_length'] = df['Text'].apply(len)
    df['average_word_length'] = df['Text'].apply(lambda x: np.mean([len(word) for word in x]) if x else 0)

    # Combine emotion and statistical features into a single DataFrame
    feature_df = df[['compound', 'text_length', 'average_word_length']]

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

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


