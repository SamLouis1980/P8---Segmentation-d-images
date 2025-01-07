from fastapi import FastAPI, File, UploadFile, Response, Query
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from model_loader import load_model, MODEL_PATHS
from io import BytesIO
from PIL import Image
import uvicorn

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
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)

    # Charger le modèle sélectionné
    model = load_model(model_name)
    
    # Prédiction
    prediction = model.predict(image_array)
    mask = np.argmax(prediction, axis=-1)[0]
    color_mask = colorize(mask)

    # Convertir en PNG
    _, buffer = cv2.imencode(".png", color_mask)
    return Response(buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
