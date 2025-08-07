#  Segmentation d’images – Système embarqué pour voiture autonome

## Contexte

L’entreprise **Future Vision Transport** conçoit des systèmes de vision par ordinateur pour véhicules autonomes.  
Dans ce cadre, l’équipe R&D travaille sur un pipeline modulaire composé de plusieurs briques fonctionnelles.  
Ce projet concerne spécifiquement la **brique de segmentation d’images**, qui reçoit en entrée des images traitées par le module précédent et doit produire des **masks de segmentation** pour alimenter la prise de décision.

Le modèle développé doit être **performant, optimisé et industrialisable**, avec une API utilisable par le système de décision, et une interface pour démontrer les résultats au reste de l’équipe.

## Objectif

- Concevoir un **modèle de segmentation d’images** entraîné sur les **8 classes principales** du jeu de données Cityscapes
- Optimiser les performances à l’aide de **data augmentation** et d’une **fonction de perte sur mesure**
- Intégrer le modèle dans une **API de prédiction** (Flask ou FastAPI) exposée sur le Cloud
- Créer une **application Streamlit** permettant de tester l’API et de visualiser les images, les masks réels et les prédictions
- Documenter la démarche dans une **note technique** et une **présentation projet**

## Technologies utilisées

- `tensorflow`, `keras`, `segmentation_models`, `albumentations`
- `MobileNetV2`, `DiceLoss`, `IOUScore`, `FScore`
- `Streamlit`, `Flask` ou `FastAPI`
- `MLFlow`, `cv2`, `PIL`, `matplotlib`, `sklearn`
- `jaccard_score`, `MeanIoU`, `train_test_split`

## Structure du projet

- notebook_principal_modelisation.ipynb - Entraînement du modèle principal
- notebook_annexe_data_augmentation.ipynb - Test des stratégies d’augmentation de données
- notebook_annexe_fonction_de_perte.ipynb - Étude des fonctions de perte adaptées
- API.py - API de prédiction déployée dans le Cloud
- streamlit.py - Interface de test de l’API et visualisation
- note_technique.pdf - Note technique (~10 pages)
- presentation.pptx - Présentation synthétique de la démarche
