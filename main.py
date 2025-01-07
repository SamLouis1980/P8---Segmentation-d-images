from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from model_loader import model
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI()

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    """
    Endpoint qui prend une image en entrée, applique la segmentation et retourne le masque prédictif.
    """
    # Charger l'image
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    image = image.resize((256, 256))  # Redimensionner à la taille attendue
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch

    # Prédiction
    prediction = model.predict(image_array)
    mask = np.argmax(prediction, axis=-1)[0]  # Récupérer la classe prédite par pixel

    # Convertir le masque en image
    mask_image = (mask * 32).astype(np.uint8)  # Remap pour affichage
    _, buffer = cv2.imencode(".png", mask_image)

    return {"filename": file.filename, "mask": buffer.tobytes()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)