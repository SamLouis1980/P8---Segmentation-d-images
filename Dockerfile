# Utiliser une image Python légère
FROM python:3.10-slim

# Installer les dépendances système depuis packages.txt
COPY packages.txt .  # Copier le fichier packages.txt dans l'image
RUN apt-get update && xargs -a packages.txt apt-get install -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Définir le port pour Cloud Run
ENV PORT=8080

# Exposer le port
EXPOSE 8080

# Lancer l'API avec le port défini
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
