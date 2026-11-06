from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the saved model bundle once, when the server starts
bundle = joblib.load(r"E:\algal bloom prediction\archive\bloom_model.pkl")
model = bundle["model"]
feature_cols = bundle["feature_cols"]
median_fill = bundle["median_fill_values"]
threshold = bundle["threshold"]

@app.route("/")
def home():
    return "Algal Bloom Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Build a single-row dataframe from the incoming data
    row = {}
    for col in feature_cols:
        if col in data and data[col] is not None:
            row[col] = data[col]
        else:
            row[col] = median_fill.get(col, 0)  # fill missing or null fields with training median

    df = pd.DataFrame([row])[feature_cols]

    prob = model.predict_proba(df)[0, 1]
    is_bloom_risk = bool(prob >= threshold)

    return jsonify({
        "bloom_probability": float(prob),
        "bloom_risk": is_bloom_risk,
        "threshold_used": threshold
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)