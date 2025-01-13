import tensorflow as tf
from google.cloud import storage
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dropout
from segmentation_models import Unet
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Désactiver CUDA pour forcer le CPU si aucun GPU n'est disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

IMAGE_PATHS = "images/RGB/"
MASK_PATHS = "images/masques/"

# Vérifie si la variable d'environnement est déjà définie
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/ayden/P8---Segmentation-d-images/cle_gcp.json"

# Mapping des tailles d'entrée des modèles
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
    """Liste toutes les images disponibles dans le bucket."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=IMAGE_PATHS)
    return [blob.name.split('/')[-1] for blob in blobs]

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un fichier depuis GCP."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_model(model_name="unet_mini"):
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join(os.getcwd(), model_path)

    if not os.path.exists(local_model_path):
        download_file(BUCKET_NAME, model_path, local_model_path)

    # Gérer les couches spécifiques à certains modèles
    custom_objects = {}
    
    if model_name == "unet_efficientnet":
        class FixedDropout(Dropout):
            def __init__(self, rate, **kwargs):
                super().__init__(rate, **kwargs)
            def call(self, inputs, training=None):
                return super().call(inputs, training=True)  # Toujours actif
        custom_objects["FixedDropout"] = FixedDropout

    # Charger le modèle avec les objets personnalisés
    model = tf.keras.models.load_model(local_model_path, compile=False, custom_objects=custom_objects)

    print(f"Modèle {model_name} chargé avec succès !")
    return model

def predict_image(model_name, image_name):
    """Effectue la prédiction sur une image choisie et applique la palette."""
    model = load_model(model_name)
    input_size = MODEL_INPUT_SIZES[model_name]

    local_image_path = os.path.join(os.getcwd(), image_name)
    download_file(BUCKET_NAME, IMAGE_PATHS + image_name, local_image_path)

    original_image = Image.open(local_image_path)
    original_size = original_image.size

    logging.debug(f"Modèle : {model_name}, Taille d'entrée attendue : {input_size}")
    logging.debug(f"Taille de l'image d'origine : {original_size}")

    # Vérification si l'image a déjà la bonne taille
    if original_size != input_size:
        logging.warning(
            f"L'image {image_name} a une taille incorrecte ({original_size}). "
            f"Elle sera redimensionnée à {input_size} pour correspondre au modèle {model_name}."
        )
        resized_image = original_image.resize(input_size, Image.BILINEAR)
    else:
        resized_image = original_image

    image_array = img_to_array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    logging.debug(f"Image entrée dans le modèle : {image_array.shape}")

    # Vérification finale après redimensionnement
    if image_array.shape[1:3] != input_size:
        logging.error(
            f"Problème de redimensionnement : {model_name} attend {input_size}, "
            f"mais reçu {image_array.shape[1:3]}"
        )
        raise ValueError(f"Taille incorrecte après redimensionnement : {image_array.shape[1:3]} au lieu de {input_size}")

    prediction = model.predict(image_array)
    logging.debug(f"Prédiction terminée. Shape sortie : {prediction.shape}")

    mask = np.argmax(prediction[0], axis=-1)

    # Appliquer la palette et redimensionner l'image de sortie à sa taille originale
    mask_colored = apply_cityscapes_palette(mask)
    mask_colored = mask_colored.resize(original_size, Image.NEAREST)

    output_path = os.path.join(os.getcwd(), image_name.replace('.png', '_pred.png'))
    mask_colored.save(output_path)

    logging.info(f"Masque prédictif sauvegardé : {output_path}")
    return output_path

if __name__ == "__main__":
    # Récupération des images disponibles dans le bucket
    available_images = list_images()

    if not available_images:
        print("Aucune image trouvée dans le bucket.")
        exit()

    # Affichage des modèles disponibles
    print("\nModèles disponibles :")
    for model in MODEL_PATHS.keys():
        print(f"- {model}")

    # Sélection du modèle par l'utilisateur
    while True:
        selected_model = input("\nEntrez le nom du modèle : ").strip()
        if selected_model in MODEL_PATHS:
            break
        print("Modèle invalide, veuillez entrer un nom correct.")

    # Affichage des images disponibles
    print("\nImages disponibles :")
    for img in available_images:
        print(f"- {img}")

    # Sélection de l'image par l'utilisateur
    while True:
        selected_image = input("\nEntrez le nom de l'image : ").strip()
        if selected_image in available_images:
            break
        print("Image invalide, veuillez entrer un nom correct.")

    # Lancement de la prédiction
    print(f"\n Lancement de la prédiction avec {selected_model} sur {selected_image}...")
    predict_image(selected_model, selected_image)
    
    predict_image(selected_model, selected_image)
