from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# On charge ton modèle .h5
model = tf.keras.models.load_model("crash_model.h5")

# On définit ce que Lovable doit envoyer (ex: les valeurs du capteur)
class CrashData(BaseModel):
    inputs: list # Une liste de chiffres

@app.get("/")
def home():
    return {"message": "L'API de Crash est prête !"}

@app.post("/predict")
def predict(data: CrashData):
    # Transformer la liste en tableau pour l'IA
    input_array = np.array([data.inputs])
    
    # Faire la prédiction
    prediction = model.predict(input_array)
    
    return {"resultat": float(prediction[0][0])}