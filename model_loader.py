import tensorflow as tf
from google.cloud import storage
import os

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "unet_mini": "models/unet_mini.h5",
    "fpn_efficientnet": "models/fpn_efficientnetb3.h5",
    "deeplabv3": "models/deeplabv3.h5"
}

# Définir l'authentification GCP avec la clé JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cle_gcp.json"

# Téléchargement du modèle depuis Google Cloud
def download_model(model_name):
    """
    Télécharge un modèle depuis Google Cloud Storage et le stocke localement.
    """
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modèle inconnu : {model_name}. Choisissez parmi {list(MODEL_PATHS.keys())}")
    
    local_path = MODEL_PATHS[model_name]
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(local_path)
    
    os.makedirs("models", exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Modèle {model_name} téléchargé avec succès depuis {BUCKET_NAME} !")
    return local_path

# Chargement du modèle
def load_model(model_name="unet_mini"):
    """
    Charge le modèle sélectionné depuis le fichier téléchargé.
    """
    model_path = download_model(model_name)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Modèle {model_name} chargé avec succès !")
    return model

# Modèle par défaut chargé
model = load_model()
    """
    model_path = download_model(model_name)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Modèle {model_name} chargé avec succès !")
    return model

# Modèle par défaut chargé
model = load_model()
