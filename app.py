import streamlit as st
import requests
import os
import logging
import model_loader
from model_loader import list_images, MODEL_PATHS, download_file, BUCKET_NAME
from PIL import Image
import cv2
import numpy as np
import importlib
from io import BytesIO

# Recharger model_loader pour éviter d'anciennes versions en cache
importlib.reload(model_loader)

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# URL de l'API FastAPI déployée sur Cloud Run
API_URL = "https://p8-deploiement-481199201103.europe-west1.run.app/predict/"

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

        # Vérification du format de l'image avant l'envoi
        image_test = Image.open(image_path)
        logging.info(f"Format de l'image envoyée : {image_test.format}")

        # Convertir l'image en PNG si nécessaire
        if image_test.format not in ["JPEG", "PNG"]:
            image_test = image_test.convert("RGB")
            image_test.save(image_path, format="PNG")
            logging.info("Image convertie en PNG avant l'envoi.")

        # Téléchargement du masque réel
        mask_real_name = selected_image.replace('_leftImg8bit.png', '_gtFine_color.png')
        mask_real_path = os.path.join(temp_dir, mask_real_name)
        try:
            download_file(BUCKET_NAME, f"images/masques/{mask_real_name}", mask_real_path)
            logging.info(f"Masque réel téléchargé : {mask_real_path}")
        except Exception as e:
            st.warning(f"Impossible de télécharger le masque réel : {e}")

        # Envoi de l'image à l'API
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            params = {"model_name": selected_model}
            response = requests.post(API_URL, params=params, files=files)

        # Log de la réponse de l'API
        logging.info(f"Réponse brute de l'API : {response.status_code} - {response.headers.get('Content-Type')}")

        # Vérification de la réponse de l'API avant de traiter l'image
        file_size = len(response.content)
        logging.info(f"Taille du fichier reçu de l'API : {file_size} octets")
        mask_pred = None
        if response.status_code == 200 and response.headers.get("Content-Type") == "image/png":
            try:
                mask_pred = Image.open(BytesIO(response.content))
                logging.info("Masque prédictif reçu et affiché directement.")

                # Redimensionner le masque prédictif
                original_image = Image.open(image_path)
                mask_pred = mask_pred.resize(original_image.size, Image.NEAREST)
                logging.info(f"Redimensionnement du masque à la taille : {original_image.size}")

                # Vérification et conversion en RGB si nécessaire
                if mask_pred.mode != "RGB":
                    mask_pred = mask_pred.convert("RGB")
                    logging.info("Masque converti en RGB.")

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
            try:
                mask_real = Image.open(mask_real_path)
                st.image(mask_real, caption="Masque Réel", width=250)
            except Exception as e:
                st.warning(f"Erreur affichage masque réel : {e}")

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
