import flask
from flask import Flask, request, jsonify
import joblib 
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data sent to this endpoint
    df = pd.DataFrame(data, index=[0])  # Convert data to a pandas DataFrame
    prediction = model.predict(df)  # Predict using your trained model
    return jsonify(prediction.tolist())  # Return the prediction as JSON

if __name__ == '__main__':
    model = joblib.load('C:\\Users\\Amin\\Desktop\\decision_tree_model.pkl')  # Load your trained model
    app.run(port=5000)  # Start the Flask app