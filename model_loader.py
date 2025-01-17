import tensorflow as tf
from google.cloud import storage
import os
import logging
import json
from tensorflow.keras.layers import Dropout
from segmentation_models import Unet
from PIL import Image

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Désactiver CUDA pour forcer le CPU si aucun GPU n'est disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Chemin de la clé GCP
GCP_CREDENTIALS_PATH = os.path.join(os.getcwd(), "cle_gcp.json")

# Vérification de la clé GCP et configuration
if os.path.exists(GCP_CREDENTIALS_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
    logging.info(f"Clé GCP chargée depuis le fichier local : {GCP_CREDENTIALS_PATH}")
else:
    logging.error("Aucune clé GCP disponible. Vérifiez le fichier local.")

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "unet_mini": "unet_mini_final.h5",
    "unet_efficientnet": "unet_efficientnet_final.h5",
    "unet_resnet34": "unet_resnet34_final.h5",
    "pspnet": "pspnet_final.h5",
    "deeplab": "deeplab_resnet50_final.h5",
    "fpn": "fpn_resnet50_final.h5"
}

MODEL_INPUT_SIZES = {
    "unet_mini": (256, 256),
    "unet_efficientnet": (256, 256),
    "unet_resnet34": (256, 256),
    "pspnet": (288, 288),
    "deeplab": (256, 256),
    "fpn": (512, 512)
}

# Palette de couleurs pour affichage
GROUP_PALETTE = [
    (0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153),
    (107, 142, 35), (70, 130, 180), (220, 20, 60), (0, 0, 142)
]

def apply_cityscapes_palette(group_mask):
    pil_mask = Image.fromarray(group_mask.astype('uint8'))
    flat_palette = [value for color in GROUP_PALETTE for value in color]
    pil_mask.putpalette(flat_palette)
    return pil_mask.convert("RGB")

def list_images():
    """Liste toutes les images disponibles dans le bucket et affiche un log détaillé."""
    try:
        logging.info("Connexion à Google Cloud Storage...")
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="images/RGB/")

        image_files = []
        for blob in blobs:
            logging.debug(f"Objet trouvé dans le bucket : {blob.name}")
            if blob.name.endswith(".png"):
                image_files.append(blob.name.split("/")[-1])

        if not image_files:
            logging.warning("Aucune image trouvée dans le dossier RGB du bucket.")
        else:
            logging.info(f"Images trouvées : {image_files}")

        return image_files
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des images : {e}")
        return []

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un fichier depuis GCP."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Fichier téléchargé : {destination_file_name}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {source_blob_name} : {e}")
        raise RuntimeError(f"Impossible de télécharger le fichier {source_blob_name}")

def load_model(model_name="unet_mini"):
    """Charge un modèle de segmentation."""
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join(os.getcwd(), model_path)

    # Vérifie si le modèle est local
    if not os.path.exists(local_model_path):
        logging.info(f"Le modèle {model_name} n'est pas trouvé localement. Tentative de téléchargement...")
        download_file(BUCKET_NAME, model_path, local_model_path)
        if not os.path.exists(local_model_path):
            raise RuntimeError(f"Le modèle {model_name} n'a pas été correctement téléchargé ou est introuvable.")
        logging.info(f"Modèle {model_name} téléchargé avec succès.")

    # Chargement du modèle
    try:
        custom_objects = {}
        if model_name == "unet_efficientnet":
            class FixedDropout(Dropout):
                def __init__(self, rate, **kwargs):
                    super().__init__(rate, **kwargs)
                def call(self, inputs, training=None):
                    return super().call(inputs, training=True)  # Toujours actif
            custom_objects["FixedDropout"] = FixedDropout

        model = tf.keras.models.load_model(local_model_path, compile=False, custom_objects=custom_objects)
        logging.debug(f"Modèle {model_name} chargé avec succès.")
        return model

    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}")
