from fastapi import FastAPI, File, UploadFile, Response, Query
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from model_loader import load_model, MODEL_PATHS, MODEL_INPUT_SIZES
from io import BytesIO
from PIL import Image
import uvicorn
import os
import logging

# Configuration du logging pour afficher les logs DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Affichage dans la console
    ]
)

logging.debug("Logging DEBUG activé !")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Liste des modèles disponibles
AVAILABLE_MODELS = list(MODEL_PATHS.keys())

# Palette officielle Cityscapes
CLASS_COLORS = {
    0: (0, 0, 0),        # Void
    1: (128, 64, 128),   # Flat
    2: (70, 70, 70),     # Construction
    3: (153, 153, 153),  # Object
    4: (107, 142, 35),   # Nature
    5: (70, 130, 180),   # Sky
    6: (220, 20, 60),    # Human
    7: (0, 0, 142)       # Vehicle
}

def colorize(mask):
    """Applique des couleurs aux classes segmentées selon la palette Cityscapes."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

@app.post("/predict/")
def predict(file: UploadFile = File(...), model_name: str = Query("unet_mini", enum=AVAILABLE_MODELS)):
    """Endpoint qui prend une image en entrée, applique la segmentation et retourne le masque colorisé."""

    # Vérification du format et de la taille
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Format non supporté. Utilisez JPEG ou PNG."}
    if file.size > 10 * 1024 * 1024:
        return {"error": "Image trop grande. Taille max: 10MB."}

    # Charger l'image
    image = Image.open(BytesIO(file.file.read())).convert("RGB")

    # Obtenir la taille d'entrée du modèle sélectionné
    if model_name not in MODEL_INPUT_SIZES:
        return {"error": f"Modèle inconnu {model_name}. Modèles disponibles : {AVAILABLE_MODELS}"}

    input_size = MODEL_INPUT_SIZES[model_name]  # Taille correcte du modèle

    logging.debug(f"[DEBUG] Modèle sélectionné : {model_name}")
    logging.debug(f"[DEBUG] Taille d'entrée attendue : {input_size}")
    logging.debug(f"[DEBUG] Taille originale de l'image : {image.size}")

    # Redimensionner l'image à la taille d'entrée du modèle
    image = image.resize(input_size, Image.BILINEAR)
    logging.debug(f"[DEBUG] Taille après redimensionnement : {image.size}")

    # Préparer l'image pour le modèle
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
    logging.debug(f"[DEBUG] Shape du tenseur avant prédiction : {image_array.shape}")

    # Charger le modèle
    model = load_model(model_name)
    
    # Vérification de la taille d'entrée du modèle
    if image_array.shape[1:3] != input_size:
        logging.error(f"[ERREUR] Taille de l'image incorrecte : {image_array.shape[1:3]}, attendu : {input_size}")
        return {"error": f"Taille incorrecte : {image_array.shape[1:3]} au lieu de {input_size}"}

    # Prédiction
    prediction = model.predict(image_array)
    logging.debug(f"[DEBUG] Prédiction terminée. Shape sortie : {prediction.shape}")

    # Extraction du masque et application de la palette
    mask = np.argmax(prediction[0], axis=-1)
    color_mask = colorize(mask)

    # Convertir en image PNG
    _, buffer = cv2.imencode(".png", color_mask)

    logging.info(f"[INFO] Prédiction réussie, image renvoyée au client.")

    return Response(buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
