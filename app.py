import streamlit as st
import requests
import os
import logging
import model_loader
from model_loader import list_images, MODEL_PATHS, download_file, BUCKET_NAME
from utils import resize_and_colorize_mask  # Import du post-traitement
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# URL de l'API FastAPI déployée sur Cloud Run
API_URL = "https://segmentation-api-481199201103.europe-west1.run.app/predict"

# Définition du répertoire temporaire (Windows/Linux)
temp_dir = os.path.join(os.getcwd(), "temp") if os.name == "nt" else "/tmp"
os.makedirs(temp_dir, exist_ok=True)

# Titre de l'application
st.title("Future Vision Transport App")
st.write("Sélectionnez un modèle et une image pour effectuer la segmentation.")

# Sélection du modèle
model_options = list(MODEL_PATHS.keys())
selected_model = st.selectbox("Choisissez un modèle :", model_options)

# Récupération des images disponibles depuis GCP
st.write("Récupération des images disponibles depuis Google Cloud Storage...")
try:
    available_images = list_images()
    logging.info(f"Images récupérées : {available_images}")
    if not available_images:
        logging.error("Aucune image trouvée dans le bucket GCP.")
        available_images = ["Aucune image disponible"]
except Exception as e:
    logging.error(f"Erreur lors de la récupération des images : {str(e)}")
    available_images = ["Erreur lors du chargement des images"]

st.write(f"Images trouvées : {available_images}")

# Sélection de l'image
selected_image = st.selectbox("Choisissez une image :", available_images)

# Vérification avant de continuer
if selected_image in ["Aucune image disponible", "Erreur lors du chargement des images"]:
    st.error("Aucune image disponible. Vérifiez la connexion au bucket GCP.")
    st.stop()

# Bouton de prédiction
if st.button("Lancer la segmentation"):
    if selected_model and selected_image:
        st.write(f"Lancement de la prédiction avec **{selected_model}** sur **{selected_image}**...")

        # Téléchargement de l'image originale
        image_path = os.path.join(temp_dir, selected_image)
        try:
            download_file(BUCKET_NAME, f"images/RGB/{selected_image}", image_path)
            logging.info(f"Image originale téléchargée : {image_path}")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de l'image : {e}")
            st.stop()

        # Téléchargement du masque réel
        real_mask_path = os.path.join(temp_dir, selected_image.replace("_leftImg8bit.png", "_gtFine_color.png"))
        try:
            download_file(BUCKET_NAME, f"images/masques/{selected_image.replace('_leftImg8bit.png', '_gtFine_color.png')}", real_mask_path)
            logging.info(f"Masque réel téléchargé : {real_mask_path}")
            real_mask = Image.open(real_mask_path)
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement du masque réel : {e}")
            real_mask = None

        # Envoi de l'image à l'API
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            params = {"model_name": selected_model}
            response = requests.post(API_URL, params=params, files=files)

        # Log de la réponse de l'API
        logging.info(f"Réponse brute de l'API : {response.status_code} - {response.headers.get('Content-Type')}")

        # Vérification de la réponse de l'API
        if response.status_code == 200 and response.headers.get("Content-Type") == "image/png":
            try:
                mask_pred = Image.open(BytesIO(response.content))
                logging.info("Masque prédictif reçu et affiché directement.")
            except Exception as e:
                logging.error(f"Erreur lors de l’ouverture du masque prédit : {e}")
                mask_pred = None
        else:
            logging.error(f"Erreur API : {response.status_code} - {response.text}")
            st.error(f"Erreur lors de la segmentation. Code erreur : {response.status_code}")

        # Affichage des résultats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            original_image = Image.open(image_path)
            st.image(original_image, caption="Image Originale", width=250)

        with col2:
            if real_mask is not None:
                st.image(real_mask, caption="Masque Réel", width=250)
            else:
                st.warning("Le masque réel n'a pas été trouvé.")

        with col3:
            if mask_pred is not None:
                st.image(mask_pred, caption="Masque Prédit", width=250)
            else:
                st.error("Le fichier du masque prédit n'a pas été généré ou est vide.")

        with col4:
            if mask_pred is not None:
                try:
                    original = np.array(original_image.convert("RGB"))
                    mask = np.array(mask_pred.convert("RGBA"))
                    mask[..., 3] = (mask[..., 3] * 0.5).astype(np.uint8)
                    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_RGB2RGBA), 1, mask, 0.6, 0)
                    st.image(Image.fromarray(overlay), caption="Superposition Masque + Image", width=250)
                except Exception as e:
                    st.error(f"Impossible d'afficher la superposition, erreur : {e}")
            else:
                st.warning("Impossible d'afficher la superposition, le masque prédit est absent.")

        st.success("Segmentation terminée avec succès !")
    else:
        st.error("Veuillez sélectionner un modèle et une image.")
