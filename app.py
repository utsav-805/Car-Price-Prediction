from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('car_price_1.pkl')

# Categorical feature mappings
fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_mapping = {'Dealer': 0, 'Individual': 1}
transmission_mapping = {'Manual': 0, 'Automatic': 1}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and parse incoming JSON data
        data = request.get_json()

        # Validate the required fields
        required_fields = ['Age', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Map categorical values to integers
        fuel_type = data['Fuel_Type']
        seller_type = data['Seller_Type']
        transmission = data['Transmission']

        if fuel_type not in fuel_mapping:
            return jsonify({'error': f'Invalid Fuel_Type: {fuel_type}'}), 400
        if seller_type not in seller_mapping:
            return jsonify({'error': f'Invalid Seller_Type: {seller_type}'}), 400
        if transmission not in transmission_mapping:
            return jsonify({'error': f'Invalid Transmission: {transmission}'}), 400

        # Prepare input data for prediction
        input_data = np.array([
            data['Age'],
            data['Present_Price'],
            data['Kms_Driven'],
            fuel_mapping[fuel_type],
            seller_mapping[seller_type],
            transmission_mapping[transmission],
            data['Owner']
        ]).reshape(1, -1)

        # Predict the car price
        predicted_price = model.predict(input_data)[0]

        # Return the predicted price
        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error if any unexpected error occurs

if __name__ == '__main__':
    app.run(debug=True)
