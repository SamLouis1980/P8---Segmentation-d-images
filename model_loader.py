def predict_image(model, image_path):
    """
    Effectue une prédiction sur une image donnée avec le modèle chargé,
    applique la palette Cityscapes et redimensionne le masque à la taille originale.
    """
    # Définition du chemin de sortie
    output_path = "/content/drive/My Drive/projet 8/segmentation_result.png"

    # Charger l'image et récupérer sa taille originale
    original_image = Image.open(image_path)
    original_size = original_image.size  # (width, height)

    # Charger et prétraiter l'image en la redimensionnant à 256x256
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Effectuer la prédiction
    prediction = model.predict(image_array)
    
    print(f"Shape de la sortie brute du modèle : {prediction.shape}")
    print(f"Valeurs uniques dans la sortie brute : {np.unique(prediction)}")

    # Transformation de la sortie en masque d'étiquettes
    mask = np.argmax(prediction[0], axis=-1)

    print(f"Shape du masque après conversion : {mask.shape}")
    print(f"Valeurs uniques dans le masque : {np.unique(mask)}")

    try:
        # Appliquer la palette de couleurs
        mask_colored = apply_cityscapes_palette(mask)

        # Redimensionner le masque à la taille originale de l'image d'entrée
        mask_colored = mask_colored.resize(original_size, Image.NEAREST)

        # Sauvegarde et affichage
        mask_colored.save(output_path)
        print(f"Masque colorisé sauvegardé dans {output_path}")

        # Affichage des résultats
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Image Originale")
        axes[0].axis("off")

        axes[1].imshow(mask_colored)
        axes[1].set_title("Masque Segmenté")
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(mask_colored, alpha=0.5)  # Superposition avec transparence
        axes[2].set_title("Superposition")
        axes[2].axis("off")

        plt.show()

    except Exception as e:
        print(f"Erreur lors de la sauvegarde du masque : {e}")
