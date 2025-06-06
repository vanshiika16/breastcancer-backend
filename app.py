from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://breastcancer-backend-wh47.onrender.com"])  # allow both local and deployed frontend

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])

    if not features or len(features) != 9:
        return jsonify({"error": "Please provide exactly 9 features"}), 400

    try:
        prediction = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features])) * 100
        return jsonify({
            "prediction": int(prediction),
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
