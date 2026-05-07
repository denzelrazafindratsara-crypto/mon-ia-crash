import numpy as np
from tensorflow.keras.models import load_model
from datetime import timedelta

model = load_model("crash_model.h5")

def predict_logic(data):

    sequence_length = 20

    if len(data) < sequence_length:
        return {
            "error": "Pas assez de données"
        }

    # =========================
    # Préparation IA
    # =========================

    last_sequence = data[-sequence_length:]

    X = np.array(last_sequence)
    X = X.reshape(1, sequence_length, 1)

    # =========================
    # Prediction IA
    # =========================

    prediction = model.predict(X)

    predicted_multiplier = float(prediction[0][0])

    # =========================
    # ETA TEMPS
    # =========================

    # Exemple :
    # 1 round = 8 secondes

    average_round_time = 8

    threshold = 10.0

    indices = [
        i for i, val in enumerate(data)
        if val >= threshold
    ]

    if len(indices) >= 2:

        gaps = [
            indices[i] - indices[i - 1]
            for i in range(1, len(indices))
        ]

        avg_gap = sum(gaps) / len(gaps)

        last_big_index = indices[-1]

        games_since_last = len(data) - 1 - last_big_index

        eta_rounds = max(
            0,
            round(avg_gap - games_since_last)
        )

        # Conversion secondes
        eta_seconds = eta_rounds * average_round_time

        # Format HH:MM:SS
        eta_time = str(
            timedelta(seconds=eta_seconds)
        )

    else:
        eta_time = "00:00:00"

    # =========================
    # Réponse API
    # =========================

    return {
        "prediction": round(predicted_multiplier, 2),
        "eta": eta_time
    }
