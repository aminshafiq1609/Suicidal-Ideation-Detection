import requests
import json

# Assuming your API is running on localhost port 5000
url = "http://127.0.0.1:5000/predict"

# Example data to send to the API
data = {
    "text": "I want to kill myself."
}

response = requests.post(url, json=data)

print(response.json())
