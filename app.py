import streamlit as st
import os
from model_loader import list_images, predict_image, MODEL_PATHS, download_file, BUCKET_NAME, MASK_PATHS
from PIL import Image

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
        output_path = predict_image(selected_model, selected_image)
        
        # Création des colonnes pour l'affichage des images
        col1, col2, col3 = st.columns(3)

        # Image originale
        with col1:
            original_path = f"/content/{selected_image}"
            original_image = Image.open(original_path)
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
        
        st.success("Segmentation terminée !")
    else:
        st.error("Veuillez sélectionner un modèle et une image.")
