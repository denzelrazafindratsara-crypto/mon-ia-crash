# mon-ia-crashimport numpy as np
from tensorflow.keras.models import load_model

# 1. Charger votre modèle .h5
model = load_model('crash_model.h5')

def predict_logic(data):
    # --- Votre logique actuelle de prédiction avec le modèle ---
    # prediction = model.predict(...)
    
    # --- NOUVELLE LOGIQUE : Calcul de l'ETA ---
    threshold = 10.0  # Le seuil pour un "gros" multiplicateur
    indices = [i for i, val in enumerate(data) if val >= threshold]
    
    if len(indices) >= 2:
        gaps = [indices[i] - indices[i-1] for i in range(1, len(indices))]
        avg_gap = sum(gaps) / len(gaps)
        last_big_index = indices[-1]
        games_since_last = len(data) - 1 - last_big_index
        eta = max(0, round(avg_gap - games_since_last))
    else:
        eta = "Données insuffisantes"
        
    return eta

# Exemple de ce que votre API doit renvoyer à Lovable
# return {"prediction": result, "eta": eta}
