# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir le port attendu par Cloud Run
ENV PORT=8080

# Exposer le port
EXPOSE 8080

# Lancer l'API avec le bon port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
