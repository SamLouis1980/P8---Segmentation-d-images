import tensorflow as tf
from google.cloud import storage
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "unet_mini": "unet_mini_final.h5",
    "unet_efficientnet": "unet_efficientnet_final.h5",
    "unet_resnet34": "unet_resnet34_final.h5",
    "pspnet": "pspnet_final.h5",
    "deeplab": "deeplab_resnet50_final.h5",
    "fpn": "fpn_final.h5"
}

IMAGE_PATHS = "images/RGB/"
MASK_PATHS = "images/masques/"

# Définir l'authentification GCP avec la clé JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/cle_gcp.json"

# Mapping des tailles d'entrée des modèles
MODEL_INPUT_SIZES = {
    "unet_mini": (256, 256),
    "unet_efficientnet": (256, 256),
    "unet_resnet34": (256, 256),
    "pspnet": (512, 512),
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
    local_model_path = f"/content/{model_path}"
    if not os.path.exists(local_model_path):
        download_file(BUCKET_NAME, model_path, local_model_path)
    model = tf.keras.models.load_model(local_model_path, compile=False)
    print(f"Modèle {model_name} chargé avec succès !")
    return model

def predict_image(model_name, image_name):
    """Effectue la prédiction sur une image choisie et applique la palette."""
    model = load_model(model_name)
    input_size = MODEL_INPUT_SIZES[model_name]
    
    local_image_path = f"/content/{image_name}"
    download_file(BUCKET_NAME, IMAGE_PATHS + image_name, local_image_path)
    
    original_image = Image.open(local_image_path)
    original_size = original_image.size
    
    image = load_img(local_image_path, target_size=input_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    mask = np.argmax(prediction[0], axis=-1)
    
    mask_colored = apply_cityscapes_palette(mask)
    mask_colored = mask_colored.resize(original_size, Image.NEAREST)
    output_path = f"/content/{image_name.replace('.png', '_pred.png')}"
    mask_colored.save(output_path)
    
    print(f"Masque prédictif sauvegardé : {output_path}")
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
