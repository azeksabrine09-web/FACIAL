import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Titre et instructions
# ---------------------------
st.title("🖼 Détection de visages avec Viola-Jones")
st.write("""
Cette application détecte les visages sur une image en utilisant l'algorithme de Viola-Jones.  
Instructions :  
1. Téléchargez une image ou prenez-en une via votre caméra.  
2. Choisissez la couleur des rectangles autour des visages.  
3. Ajustez les paramètres `scaleFactor` et `minNeighbors` pour améliorer la détection.  
4. Cliquez sur 'Détecter les visages' pour voir le résultat.  
5. Cliquez sur 'Enregistrer l'image' pour sauvegarder l'image avec les visages détectés.
""")

# ---------------------------
# Upload d'image
# ---------------------------
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))

    # Convertir en BGR pour OpenCV
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # ---------------------------
    # Choix de la couleur des rectangles
    # ---------------------------
    color = st.color_picker("Choisissez la couleur des rectangles", "#FF0000")
    # Convertir couleur hex en BGR tuple
    color_bgr = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]

    # ---------------------------
    # Paramètres minNeighbors et scaleFactor
    # ---------------------------
    scaleFactor = st.slider("Scale Factor", min_value=1.01, max_value=2.0, value=1.1, step=0.01)
    minNeighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, step=1)

    # ---------------------------
    # Détection faciale
    # ---------------------------
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

    # ---------------------------
    # Afficher l'image avec les visages détectés
    # ---------------------------
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Image avec visages détectés", use_column_width=True)

    # ---------------------------
    # Bouton pour enregistrer l'image
    # ---------------------------
    if st.button("Enregistrer l'image"):
        save_path = "visages_detectes.png"
        cv2.imwrite(save_path, frame)
        st.success(f"Image enregistrée avec succès : {save_path}")
