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
    handlers=[logging.StreamHandler()]
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
async def predict(file: UploadFile = File(...), model_name: str = Query("unet_mini", enum=AVAILABLE_MODELS)):
    """Endpoint qui prend une image en entrée, applique la segmentation et retourne le masque colorisé."""

    logging.debug(f"Requête reçue avec modèle : {model_name}")

    # Vérification du format et de la taille
    if file.content_type not in ["image/jpeg", "image/png"]:
        logging.error("Format non supporté reçu !")
        return {"error": "Format non supporté. Utilisez JPEG ou PNG."}
    
    if file.size > 10 * 1024 * 1024:
        logging.error("Image trop grande (>10MB) reçue !")
        return {"error": "Image trop grande. Taille max: 10MB."}

    # Lire l'image reçue
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            logging.error("Fichier image vide reçu !")
            return {"error": "Fichier image vide"}
        
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size  # Sauvegarde de la taille originale
        logging.debug(f"Taille originale de l'image : {original_size}")

    except Exception as e:
        logging.error(f"Impossible de lire l'image reçue : {e}")
        return {"error": "Format d'image non supporté"}

    # Vérification du modèle sélectionné
    if model_name not in MODEL_INPUT_SIZES:
        logging.error(f"Modèle inconnu {model_name} demandé !")
        return {"error": f"Modèle inconnu {model_name}. Modèles disponibles : {AVAILABLE_MODELS}"}

    input_size = MODEL_INPUT_SIZES[model_name]  # Taille correcte du modèle
    logging.debug(f"Modèle {model_name} sélectionné - Taille attendue : {input_size}")

    # Redimensionner l'image
    image = image.resize(input_size, Image.BILINEAR)
    logging.debug(f"Taille après redimensionnement : {image.size}")

    # Préparer l'image pour le modèle
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
    logging.debug(f"Shape du tenseur avant prédiction : {image_array.shape}")

    # Charger le modèle
    logging.info(f"Chargement du modèle {model_name}...")
    model = load_model(model_name)

    # Vérification de la taille d'entrée du modèle
    if image_array.shape[1:3] != input_size:
        logging.error(f"ERREUR : Taille incorrecte {image_array.shape[1:3]}, attendu {input_size}")
        return {"error": f"Taille incorrecte : {image_array.shape[1:3]} au lieu de {input_size}"}

    # Prédiction
    logging.info("Exécution de la prédiction...")
    prediction = model.predict(image_array)
    logging.debug(f"Prédiction terminée. Shape sortie : {prediction.shape}")

    # Extraction du masque et application de la palette
    mask = np.argmax(prediction[0], axis=-1)

    # Redimensionner le masque à la taille originale de l’image
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.resize(original_size, Image.NEAREST)
    logging.info(f"Masque redimensionné à la taille : {original_size}")

    # Appliquer la palette de couleurs
    color_mask = colorize(np.array(mask))

    # Vérification du masque généré
    if color_mask is None or color_mask.size == 0:
        logging.error("Le masque généré est vide !")
        return {"error": "Le masque généré est vide"}

    # Sauvegarde temporaire pour debug
    debug_path = "debug_mask_pred.png"
    cv2.imwrite(debug_path, color_mask)
    logging.info(f"Masque prédictif sauvegardé temporairement sous '{debug_path}'")

    # Convertir en image PNG
    success, buffer = cv2.imencode(".png", color_mask)

    if not success or buffer is None or len(buffer.tobytes()) == 0:
        logging.error("Échec de l'encodage du masque en PNG !")
        return {"error": "Erreur lors de l'encodage du masque prédictif"}

    logging.info(f"Masque généré avec succès ({len(buffer.tobytes())} bytes), envoi au client.")
    return Response(buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
