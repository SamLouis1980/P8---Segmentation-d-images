import streamlit as st
import requests
import os
import logging
from model_loader import list_images, MODEL_PATHS, download_file, BUCKET_NAME, MASK_PATHS
from PIL import Image
import cv2
import numpy as np
import importlib
import model_loader

# Recharger model_loader (utile si Streamlit garde des anciennes versions en cache)
importlib.reload(model_loader)

# Configuration du logging dans Streamlit
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# URL de l'API FastAPI déployée sur Cloud Run
API_URL = "https://p8-deploiement-481199201103.europe-west1.run.app/predict/"

# **Titre de l'application**
st.title("Future Vision Transport App")
st.write("Sélectionnez un modèle et une image pour effectuer la segmentation.")

# **Sélection du modèle**
model_options = list(MODEL_PATHS.keys())
selected_model = st.selectbox("Choisissez un modèle :", model_options)

# **Récupération des images disponibles dans GCP**
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

# **Affichage explicite de la liste des images**
st.write(f"Images trouvées : {available_images}")

# **Sélection de l'image**
selected_image = st.selectbox("Choisissez une image :", available_images)

# **Bloquer l'application si aucune image n'est trouvée**
if selected_image in ["Aucune image disponible", "Erreur lors du chargement des images"]:
    st.error("Aucune image n'est disponible. Vérifiez la connexion au bucket GCP.")
    st.stop()

# **Bouton de prédiction**
if st.button("Lancer la segmentation"):
    if selected_model and selected_image:
        st.write(f"Lancement de la prédiction avec **{selected_model}** sur **{selected_image}**...")

        # **Téléchargement de l'image depuis GCP**
        image_path = f"/tmp/{selected_image}"  # Changer le chemin vers /tmp/ pour Streamlit Cloud
        try:
            download_file(BUCKET_NAME, f"images/RGB/{selected_image}", image_path)
            logging.info(f"Image téléchargée : {image_path}")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de l'image : {e}")
            st.stop()

        # **Envoi de l'image à l'API**
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            params = {"model_name": selected_model}
            response = requests.post(API_URL, params=params, files=files)

        # **Traitement de la réponse de l'API**
        if response.status_code == 200:
            output_path = "/tmp/mask_pred.png"
            with open(output_path, "wb") as f:
                f.write(response.content)
            logging.info("Masque prédictif reçu et sauvegardé.")
        else:
            st.error("Erreur lors de la segmentation !")
            st.write(f"Code erreur : {response.status_code}")
            st.write(response.text)
            output_path = None

        # **Affichage des résultats**
        if output_path:
            col1, col2, col3 = st.columns(3)

            # **Image originale**
            with col1:
                original_image = Image.open(image_path)
                st.image(original_image, caption="Image Originale", use_container_width=True)

            # **Masque réel**
            with col2:
                mask_real_name = selected_image.replace('_leftImg8bit.png', '_gtFine_color.png')
                mask_real_path = f"/tmp/{mask_real_name}"

                # **Téléchargement du masque réel depuis GCP**
                try:
                    download_file(BUCKET_NAME, f"images/masques/{mask_real_name}", mask_real_path)
                    mask_real = Image.open(mask_real_path)
                    st.image(mask_real, caption="Masque Réel", use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossible de télécharger le masque réel : {e}")

            # **Masque prédit**
            with col3:
                mask_pred = Image.open(output_path)
                st.image(mask_pred, caption="Masque Prédit", use_container_width=True)

            # **Superposition du masque prédict sur l'image originale**
            st.write("### Superposition du masque prédict sur l'image originale")
            original = np.array(original_image.convert("RGB"))  # Convertir PIL -> NumPy
            mask = np.array(mask_pred.convert("RGBA"))  # Convertir en image RGBA

            # Appliquer la transparence sur le masque (alpha = 0.5)
            alpha = 0.5
            mask[..., 3] = (mask[..., 3] * alpha).astype(np.uint8)

            # Convertir les images pour OpenCV
            original_cv = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
            mask_cv = mask

            # Superposition avec addWeighted
            overlay = cv2.addWeighted(original_cv, 1, mask_cv, 0.6, 0)

            # Convertir en image PIL et afficher
            overlay_pil = Image.fromarray(overlay)
            st.image(overlay_pil, caption="Superposition Masque + Image", use_container_width=True)

            st.success("Segmentation terminée avec succès !")
    else:
        st.error("Veuillez sélectionner un modèle et une image.")
