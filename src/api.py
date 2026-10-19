
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "attrition_model.joblib"

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        employee = pd.DataFrame([data])[FEATURES]

        prediction = model.predict(employee)[0]

        result = {
            "prediction": int(prediction)
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(employee)[0]
            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(model.classes_, probabilities)
            }

        return jsonify(result)

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
