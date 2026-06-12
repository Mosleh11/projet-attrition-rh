"""API Flask : prédire le risque de départ d'un employé.

Lancement :
    pip install flask
    python api.py
Puis : POST http://localhost:5000/predict  avec {"features": [ ... 47 nombres ... ]}

Auteur de cette partie : Chaimae IMRANI
"""
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
BUNDLE = joblib.load("modele.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "cle 'features' manquante"}), 400

    features = data["features"]
    attendu = len(BUNDLE["features"])
    if not isinstance(features, list) or len(features) != attendu:
        return jsonify({"error": f"il faut {attendu} features numeriques"}), 400

    try:
        x = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({"error": "les features doivent etre des nombres"}), 400

    x_s = BUNDLE["scaler"].transform(x)
    pred = int(BUNDLE["modele"].predict(x_s)[0])
    proba = float(BUNDLE["modele"].predict_proba(x_s)[0][1])
    reponse = {
        "prediction": pred,
        "proba_depart": round(proba, 3),
        "label": "risque de depart" if pred == 1 else "va probablement rester",
    }
    return jsonify(reponse), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
