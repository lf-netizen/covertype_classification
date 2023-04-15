from flask import Flask, request
import tensorflow as tf
import pickle
import numpy as np

from covertype_classification import NAMES_CAT, NAMES_CONT, MODEL_SAVEDIR, simple_heuristic

with open(MODEL_SAVEDIR + 'RandomForest.pkl', 'rb') as f:
    rf = pickle.load(f)

with open(MODEL_SAVEDIR + 'KNeighbors.pkl', 'rb') as f:
    neigh = pickle.load(f)

nn = tf.keras.models.load_model(MODEL_SAVEDIR + 'NeuralNet')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    """
    Endpoint for predicting the forest cover type using machine learning models.

    Accepts a POST request with a JSON payload containing the following keys:
    - model: the name of the machine learning model to use for prediction (string)
    - features: a dictionary of input features in the following format:
        - Elevation (float)
        - Aspect (float)
        - Slope (float)
        - Horizontal_Distance_To_Hydrology (float)
        - Vertical_Distance_To_Hydrology (float)
        - Horizontal_Distance_To_Roadways (float)
        - Hillshade_9am (float)
        - Hillshade_Noon (float)
        - Hillshade_3pm (float)
        - Horizontal_Distance_To_Fire_Points (float)
        - Wilderness_Area (integer): an integer in the range [1, 4] corresponding to the type of wilderness area
        - Soil_Type (integer): an integer in the range [1, 40] corresponding to the type of soil
        
    Example input:
    {
        'model': 'NeuralNetwork',
        'features': {
            'Elevation': 3351.0,
            'Aspect': 206.0,
            'Slope': 27.0,
            'Horizontal_Distance_To_Hydrology': 726.0,
            'Vertical_Distance_To_Hydrology': 124.0,
            'Horizontal_Distance_To_Roadways': 3813.0,
            'Hillshade_9am': 192.0,
            'Hillshade_Noon': 252.0,
            'Hillshade_3pm': 180.0,
            'Horizontal_Distance_To_Fire_Points': 2271.0,
            'Wilderness_Area': 1,
            'Soil_Type': 38
        }
    }
    
    Returns a JSON object with a single key:
    - prediction: the predicted forest cover type as an integer in the range [1, 7]
    """
    model_name = request.json['model']
    features = request.json['features']

    # one-hot encode
    features_Wilderness_Area = {f'Wilderness_Area{i+1}':int(i+1 == features['Wilderness_Area']) for i in range(4)}
    features_Soil_Type = {f'Soil_Type{i+1}':int(i+1 == features['Soil_Type']) for i in range(40)}
    features = {**features, **features_Wilderness_Area, **features_Soil_Type}
    
    # convert into vector
    feature_vec = [features[name] for name in NAMES_CONT + NAMES_CAT]
    feature_vec = np.array(feature_vec).reshape(1, -1)

    # predict
    if model_name == 'SimpleHeuristic':
        pred = simple_heuristic(feature_vec)
    elif model_name == 'RandomForest':
        pred = rf.predict(feature_vec)
    elif model_name == 'KNeighbors':
        pred = neigh.predict(feature_vec)
    elif model_name == 'NeuralNetwork':
        pred = np.argmax(nn.predict(feature_vec), axis=1)
    else:
        return {'prediction': f'Unknown model name: {model_name}.'}

    return {'prediction': int(pred[0]) + 1} # + 1 because we initially subtracted 1 from all the classes

if __name__ == '__main__':
    app.run()