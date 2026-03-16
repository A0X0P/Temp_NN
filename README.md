# Indoor Classroom Temperature Prediction Using A Neural Network

## 1. Project Overview
This project contains a Deep Learning Neural Network trained using TensorFlow & Keras. Its purpose is to predict the localized indoor temperature of specific grids within a classroom based on outside weather, room physics (height elevation), and occupancy dynamics (people, fans, open windows).

## 2. Model Architecture
The model is a fully connected Feedforward Neural Network:
* **Input Layer:** Accepts 12 scaled and encoded features.
* **Hidden Layer 1:** 64 Neurons (ReLU activation).
* **Hidden Layer 2:** 32 Neurons (ReLU activation).
* **Output Layer:** 1 Neuron (Linear output predicting Temperature in °C).

## 3. The Files
If you downloaded this project, you should have two required files to run predictions:
1. `temperature_nn_model.keras` - The trained Neural Network weights.
2. `nn_preprocessor.joblib` - The Scikit-Learn scaler/encoder needed to format new raw data for the neural network.

## 4. How to Use the Saved Model in Python
To make predictions on new data, use the following code snippet:

```python
import pandas as pd
import joblib
from tensorflow import keras

# 1. Load the preprocessor and the model
preprocessor = joblib.load('nn_preprocessor.joblib')
model = keras.models.load_model('temperature_nn_model.keras')

# 2. Define your new scenario
new_data = pd.DataFrame({
    'Hour': [15],
    'Grid Row': ['B'],
    'Grid Column': ['G2'],
    'People': [4],
    'Windows Open': [0],
    'AC/Fan On': [1],
    'No of AC/Fans': [2],
    'No of Devices': [1],
    'Height': [3.0],
    'Outside Temp (°C)': [35.0],
    'Outside Humidity (%)': [55.0],
    'Precipitation': [0]
})

# 3. Preprocess the data and Predict!
processed_data = preprocessor.transform(new_data)
prediction = model.predict(processed_data)

print(f"Predicted Temp: {prediction[0][0]:.2f} °C")
