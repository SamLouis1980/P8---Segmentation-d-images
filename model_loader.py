from google.cloud import storage
import os
import logging
import json
from PIL import Image
import toml
import streamlit as st
import torch
from transformers import Mask2FormerForUniversalSegmentation

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Désactiver CUDA pour forcer le CPU si aucun GPU n'est disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Charger la clé GCP depuis les secrets Streamlit
try:
    if "GCP_CREDENTIALS" in st.secrets:
        # Récupérer les secrets via Streamlit
        credentials_json = st.secrets["GCP_CREDENTIALS"]
        credentials_dict = json.loads(credentials_json) if isinstance(credentials_json, str) else credentials_json

        # Créer un fichier temporaire pour la clé
        GCP_CREDENTIALS_PATH = "/tmp/gcp_key.json"
        with open(GCP_CREDENTIALS_PATH, "w") as f:
            json.dump(credentials_dict, f)

        # Vérifier si le fichier a bien été créé
        with open(GCP_CREDENTIALS_PATH, "r") as f:
            logging.info(f"Contenu du fichier clé GCP : {f.read()}")

        # Définir la variable d'environnement pour GCP
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
        logging.info("✅ Clé GCP chargée depuis les secrets Streamlit Cloud.")
    else:
        logging.error("❌ Aucune clé GCP trouvée dans Streamlit Secrets.")
        raise RuntimeError("Erreur de configuration GCP. Vérifiez les secrets Streamlit.")
except json.JSONDecodeError as e:
    logging.error(f"❌ Erreur de décodage JSON dans GCP_CREDENTIALS : {e}")
    raise RuntimeError("Erreur de décodage JSON dans les secrets GCP.")
except Exception as e:
    logging.error(f"❌ Impossible de charger la clé GCP depuis les secrets : {e}")
    raise RuntimeError("Erreur de configuration GCP. Vérifiez les secrets Streamlit.")

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "fpn": "fpn_resnet50_final.h5",
    "mask2former": "mask2former_final.h5"
}

MODEL_INPUT_SIZES = {
    "fpn": (512, 512),
    "mask2former": (512, 512)
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

def load_model(model_name="fpn"):
    """Charge un modèle de segmentation basé sur Torch."""
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join(os.getcwd(), model_path)

    # Vérifie si le modèle est local
    if not os.path.exists(local_model_path):
        logging.info(f"Le modèle {model_name} n'est pas trouvé localement. Tentative de téléchargement...")
        download_file(BUCKET_NAME, model_path, local_model_path)
        if not os.path.exists(local_model_path):
            raise RuntimeError(f"Le modèle {model_name} n'a pas été correctement téléchargé ou est introuvable.")
        logging.info(f"Modèle {model_name} téléchargé avec succès.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "mask2former":
            model = Mask2FormerForUniversalSegmentation.from_pretrained(local_model_path).to(device)
        else:
            model = torch.load(local_model_path, map_location=device)

        model.eval()  # Mode évaluation
        logging.info(f"Modèle {model_name} chargé avec succès.")
        return model

    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}")
