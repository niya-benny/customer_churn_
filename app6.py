from flask import Flask, request, render_template_string
import numpy as np
import joblib
import os
import subprocess
import pandas as pd

# Ensure model and encoders are ready
required_files = ["logreg_model.joblib", "encoders6.joblib", "selected_features.joblib"]
if not all([os.path.exists(f) for f in required_files]):
    subprocess.run(["python", "churn6.py"])

# Load model and encoders
model = joblib.load("logreg_model.joblib")
encoders = joblib.load("encoders6.joblib")  # Dict of column_name: LabelEncoder
selected_features = joblib.load("selected_features.joblib")  # List of selected features

app = Flask(__name__)

# Dropdowns (should match the training categories)
dropdown_options = {
    'SubscriptionType': ['Basic', 'Standard', 'Premium'],
    'PaymentMethod': ['Credit card', 'Bank transfer', 'Mailed check','Electronic check' 'Other'],
    'PaperlessBilling': ['Yes', 'No'],
    'ContentType': ['Movies', 'TV Shows', 'Both'],
    'MultiDeviceAccess': ['Yes', 'No'],
    'DeviceRegistered': ['TV', 'Mobile', 'Tablet', 'Computer'],
    'GenrePreference': ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Other'],
    'Gender': ['Male', 'Female'],
    'ParentalControl': ['Yes', 'No'],
    'SubtitlesEnabled': ['Yes', 'No'],
}

# Simple form template
form_template = """
<!DOCTYPE html>
<html>
<head><title>Churn Prediction</title></head>
<body>
<h2>Customer Churn Prediction</h2>
<form method="POST" action="/predict">
    {% for field in numeric_fields %}
        <label>{{ field }}:</label><br>
        <input type="number" step="any" name="{{ field }}" required><br><br>
    {% endfor %}
    {% for feature, options in dropdown_options.items() %}
        <label>{{ feature }}:</label><br>
        <select name="{{ feature }}" required>
            {% for option in options %}
                <option value="{{ option }}">{{ option }}</option>
            {% endfor %}
        </select><br><br>
    {% endfor %}
    <input type="submit" value="Predict">
</form>
</body>
</html>
"""

# Feature names to collect from form
numeric_fields = [
    'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
    'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating'
]

@app.route('/')
def home():
    return render_template_string(form_template, dropdown_options=dropdown_options, numeric_fields=numeric_fields)

@app.route('/predict', methods=['POST'])
def predict():
    input_dict = {}

    # Collect numeric fields
    for field in numeric_fields:
        try:
            input_dict[field] = float(request.form[field])
        except ValueError:
            return f"Invalid input for {field}", 400

    # Collect and encode categorical fields
    for cat_feat in dropdown_options:
        raw_val = request.form[cat_feat]
        encoder = encoders.get(cat_feat)
        if encoder is not None:
            try:
                encoded_val = encoder.transform([raw_val])[0]
                input_dict[cat_feat] = encoded_val
            except ValueError:
                return f"Unexpected value for {cat_feat}", 400
        else:
            return f"Missing encoder for {cat_feat}", 500

    # Form a DataFrame and reindex to match selected features
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=selected_features, fill_value=0)

    # Predict with threshold
    proba = model.predict_proba(input_df)[0][1]
    threshold = 0.67
    prediction = "Yes" if proba >= threshold else "No"

    return f"""
    <h2>Prediction Result</h2>
    <p>Churn Probability: {proba:.2f}</p>
    <p>Predicted Churn: {prediction}</p>
    <a href="/">Try again</a>
    """

if __name__ == '__main__':
    app.run(debug=True)
