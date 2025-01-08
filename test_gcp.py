from google.cloud import storage
import os

# Définir l'authentification explicitement
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/cle_gcp.json"

# Tester l'accès au bucket
client = storage.Client()
bucket_name = "p8_segmentation_models"

try:
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs())

    print(f"Accès réussi au bucket : {bucket_name}")
    print(f"Fichiers trouvés : {[blob.name for blob in blobs]}")

except Exception as e:
    print(f"Erreur d'accès au bucket : {e}")
