# Utiliser une image Python légère
FROM python:3.11

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Installer PyTorch AVANT les autres dépendances
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

RUN pip cache purge

# Définir le port pour Cloud Run
ENV PORT=8080

# Exposer le port
EXPOSE 8080

# Lancer l'API avec le port défini
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
