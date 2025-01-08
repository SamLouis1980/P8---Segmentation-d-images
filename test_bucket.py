from google.cloud import storage
import os

# Définir l'authentification GCP avec la clé JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cle_gcp.json"

# Nom du bucket
BUCKET_NAME = "p8_segmentation_models"

try:
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    
    # Lister les fichiers
    blobs = list(bucket.list_blobs())
    print(f"Fichiers trouvés dans {BUCKET_NAME} : {[blob.name for blob in blobs]}")

except Exception as e:
    print(f"Erreur lors de l'accès au bucket : {e}")
