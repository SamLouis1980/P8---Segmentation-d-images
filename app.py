import streamlit as st
import requests
import os
from model_loader import list_images, MODEL_PATHS, download_file, BUCKET_NAME, MASK_PATHS
from PIL import Image
import cv2
import numpy as np

# URL de l'API FastAPI déployée sur Cloud Run
API_URL = "https://p8-deploiement-481199201103.europe-west1.run.app/predict"

# Titre de l'application
st.title("Segmentation d'images - Projet 8")
st.write("Sélectionnez un modèle et une image pour effectuer la segmentation.")

# Sélection du modèle
model_options = list(MODEL_PATHS.keys())
selected_model = st.selectbox("Choisissez un modèle :", model_options)

# Sélection de l'image
available_images = list_images()
selected_image = st.selectbox("Choisissez une image :", available_images)

# Bouton de prédiction
if st.button("Lancer la segmentation"):
    if selected_model and selected_image:
        st.write(f"Lancement de la prédiction avec {selected_model} sur {selected_image}...")

        # Charger l'image sélectionnée
        image_path = f"/content/{selected_image}"
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            params = {"model_name": selected_model}

            # Envoi de la requête à l'API FastAPI
            response = requests.post(API_URL, params=params, files=files)

            if response.status_code == 200:
                # Sauvegarde de l'image retournée
                output_path = "/content/mask_pred.png"
                with open(output_path, "wb") as f:
                    f.write(response.content)
            else:
                st.error("Erreur lors de la segmentation !")
                st.write(f"Code erreur : {response.status_code}")
                st.write(response.text)
                output_path = None

        # Vérification si output_path est valide avant d'afficher l'image
        if output_path:
            # Création des colonnes pour l'affichage des images
            col1, col2, col3 = st.columns(3)

            # Image originale
            with col1:
                original_image = Image.open(image_path)
                st.image(original_image, caption="Image Originale", use_container_width=True)

            # Masque réel
            with col2:
                mask_real_name = selected_image.replace('_leftImg8bit.png', '_gtFine_color.png')
                mask_real_path = f"/content/{mask_real_name}"

                # Télécharger le masque réel depuis GCP
                download_file(BUCKET_NAME, MASK_PATHS + mask_real_name, mask_real_path)

                # Charger et afficher l'image
                mask_real = Image.open(mask_real_path)
                st.image(mask_real, caption="Masque Réel", use_container_width=True)

            # Masque prédit
            with col3:
                mask_pred = Image.open(output_path)
                st.image(mask_pred, caption="Masque Prédit", use_container_width=True)

            # Superposition du masque prédict sur l'image originale (en-dessous des colonnes)
            st.write("### Superposition du masque prédict sur l'image originale")
            original = np.array(original_image.convert("RGB"))  # Convertir PIL -> NumPy
            mask = np.array(mask_pred.convert("RGBA"))  # Convertir en image RGBA

            # Appliquer la transparence sur le masque (alpha = 0.5)
            alpha = 0.5
            mask[..., 3] = (mask[..., 3] * alpha).astype(np.uint8)  # Modifier canal alpha

            # Convertir les images pour OpenCV
            original_cv = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
            mask_cv = mask

            # Superposition avec addWeighted
            overlay = cv2.addWeighted(original_cv, 1, mask_cv, 0.6, 0)

            # Convertir en image PIL et afficher
            overlay_pil = Image.fromarray(overlay)
            st.image(overlay_pil, caption="Superposition Masque + Image", use_container_width=True)

            st.success("Segmentation terminée !")
    else:
        st.error("Veuillez sélectionner un modèle et une image.")
