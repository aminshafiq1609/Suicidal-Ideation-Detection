// Function to classify text and show the result
function classifyText(text) {
    // Make a request to your Flask application for text classification
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ Text: text })
    })
      .then(response => response.json())
      .then(prediction => {
        // Display the result in the UI
        const resultContainer = document.getElementById('result');
        const predictionText = document.getElementById('prediction-text');
        predictionText.textContent = prediction[0]; // Assuming the prediction is a single value
        resultContainer.style.display = 'block';
      })
      .catch(error => {
        console.error(error);
        // Handle any error that occurred during classification
      });
  }
  
  // Listen for input event on the textarea
  const inputText = document.getElementById('input-text');
  inputText.addEventListener('input', function() {
    const text = inputText.value;
    // Call the classifyText function with the user input
    classifyText(text);
  });
  