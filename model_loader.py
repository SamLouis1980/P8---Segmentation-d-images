import tensorflow as tf
from google.cloud import storage
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "unet_mini": "unet_mini_final.h5",
    "unet_efficientnet": "unet_efficientnet_final.h5",
    "unet_resnet34": "unet_resnet34_final.h5"
}

# Définir l'authentification GCP avec la clé JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/cle_gcp.json"

# Téléchargement du modèle depuis Google Cloud
def download_model(model_name):
    """
    Télécharge un modèle depuis Google Cloud Storage et le stocke localement.
    """
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modèle inconnu : {model_name}. Choisissez parmi {list(MODEL_PATHS.keys())}")

    local_file_path = MODEL_PATHS[model_name]  # Nom du fichier local

    try:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)

        print(f"Accès réussi au bucket : {BUCKET_NAME}")

        # Vérifier si le fichier existe avant de le télécharger
        blobs = [blob.name for blob in bucket.list_blobs()]
        
        if MODEL_PATHS[model_name] not in blobs:
            raise FileNotFoundError(f"Le fichier {MODEL_PATHS[model_name]} n'existe pas dans le bucket.")

        blob = bucket.blob(MODEL_PATHS[model_name])
        blob.download_to_filename(local_file_path)

        print(f"Modèle {model_name} téléchargé avec succès dans {local_file_path} !")

        return local_file_path

    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle {model_name} : {e}")
        raise

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

def predict_image(model, image_path):
    """
    Effectue une prédiction sur une image donnée avec le modèle chargé.
    """
    # Définition du chemin de sortie
    output_path = "/content/drive/My Drive/projet 8/segmentation_result.png"

    # Charger et prétraiter l'image
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Effectuer la prédiction
    prediction = model.predict(image_array)
    
    print(f"Shape de la sortie brute du modèle : {prediction.shape}")
    print(f"Valeurs uniques dans la sortie brute : {np.unique(prediction)}")

    # Transformation de la sortie en masque d'étiquettes
    mask = np.argmax(prediction[0], axis=-1)

    print(f"Shape du masque après conversion : {mask.shape}")
    print(f"Valeurs uniques dans le masque : {np.unique(mask)}")

    try:
        # Convertir le masque en une image lisible (échelle des valeurs 0-7 en 0-255)
        mask_image = Image.fromarray((mask * (255 / 7)).astype(np.uint8))
        mask_image.save(output_path)
        print(f"Masque sauvegardé dans {output_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du masque : {e}")

# Test de prédiction
if __name__ == "__main__":
    test_image_path = "/content/drive/My Drive/projet 8/test_image.png"
    predict_image(model, test_image_path)
