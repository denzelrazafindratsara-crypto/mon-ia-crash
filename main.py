import os
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS # Très important pour Lovable !

app = Flask(__name__)
CORS(app) # Autorise Lovable à appeler ton API sans blocage de sécurité

# Chargement sécurisé des modèles
eta_model = joblib.load("eta_model.pkl")
mult_model = joblib.load("mult_model.pkl")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # On attend une liste de nombres dans l'URL
        input_data = request.args.get('values') 
        features = [[float(x) for x in input_data.split(',')]]
        
        eta = eta_model.predict(features)[0]
        mult = mult_model.predict(features)[0]

        return jsonify({
            "status": "success",
            "eta_seconds": float(eta),
            "predicted_multiplier": float(mult)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Railway utilise la variable d'environnement PORT
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)