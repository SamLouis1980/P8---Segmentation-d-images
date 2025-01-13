# Étape 1 : Utiliser une image de base avec Python 3.10
FROM python:3.10-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers de l’application
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port utilisé par Uvicorn
EXPOSE 8000

# Étape 6 : Lancer l’API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
