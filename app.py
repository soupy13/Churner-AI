import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras

app = Flask(__name__)

# --- Load Model & Assets ---
try:
    with open('churn_model_assets.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
    # Reconstruct the Keras model from config and weights
    model = keras.Sequential.from_config(model_data['model_config'])
    model.set_weights(model_data['model_weights'])
    
    # Extract the scaler and the list of columns to scale
    scaler = model_data['scaler']
    cols_to_scale = model_data.get('cols_to_scale', [])
    
    # Retrieve the exact features the scaler (and model) expects
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model. Error: {e}")
    model, scaler, cols_to_scale, expected_features = None, None, [], None

@app.route('/')
def home():
    """Renders the frontend HTML interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint to accept user data and return a churn prediction."""
    if not model:
        return jsonify({'error': 'Model is not loaded properly on the server.'}), 500

    try:
        # 1. Receive data from frontend
        data = request.json
        
        # 2. Convert to DataFrame
        df_input = pd.DataFrame([data])
        
        # 3. Handle Categorical Data (One-Hot Encoding)
        # We dummy-encode the categorical columns just like in the notebook
        df_encoded = pd.get_dummies(df_input)
        
        # 4. Align Columns with the expected model input
        if expected_features is not None:
            # Reindex adds missing columns as NaNs, which we fill with 0
            df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)
        else:
            df_aligned = df_encoded
            
        # 5. Apply Scaling
        if cols_to_scale:
            # Scale only the designated columns
            df_aligned[cols_to_scale] = scaler.transform(df_aligned[cols_to_scale])
        else:
            # If no specific columns, attempt to scale everything
            df_aligned = pd.DataFrame(scaler.transform(df_aligned), columns=df_aligned.columns)
            
        # 6. Make Prediction
        prediction_prob = model.predict(df_aligned.values)
        churn_probability = float(prediction_prob[0][0])
        
        # 7. Format Output
        is_churn = bool(churn_probability > 0.5)
        
        return jsonify({
            'churn_probability': round(churn_probability * 100, 2),
            'prediction': 'High Risk of Churn' if is_churn else 'Low Risk of Churn',
            'is_churn': is_churn
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)