import tensorflow as tf

def load_model(model_path="unet_mini.h5"):
    """
    Charge le modèle de segmentation sauvegardé en format H5.
    
    Args:
        model_path (str): Chemin du fichier H5 du modèle.
    
    Returns:
        tf.keras.Model: Modèle chargé prêt pour la prédiction.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Modèle chargé avec succès !")
    return model

# Charger le modèle au démarrage
model = load_model()
