import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import base64
from io import BytesIO
import torch

# Cargar el modelo preentrenado EfficientNetB0 con las capas superiores
efficientnet_model = EfficientNetB0(weights='imagenet')

# Cargar el modelo y el tokenizador para la traducción
translation_model_name = 'facebook/m2m100_418M'
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)
translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)

# Función para clasificar la imagen usando EfficientNetB0
def classify_image(img):
    # Redimensionar la imagen a 224x224 píxeles como se espera por EfficientNetB0
    img = img.resize((224, 224))
    # Convertir la imagen a un array y preprocesarla
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Clasificar la imagen
    predictions = efficientnet_model.predict(img_array)
    return decode_predictions(predictions, top=3)[0]  # Decodificar las 3 principales predicciones

# Función para traducir texto al español
def traducir_texto(texto):
    # Preprocesar y traducir el texto
    translation_tokenizer.src_lang = "en"
    inputs = translation_tokenizer(texto, return_tensors="pt", truncation=True)
    generated_tokens = translation_model.generate(**inputs, forced_bos_token_id=translation_tokenizer.get_lang_id("es"))
    traduccion = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return traduccion[0]

# Función para decodificar una imagen en base64
def decodificar_imagen_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

# Cargar el modelo YOLOv5 para detección de objetos
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Función para detectar objetos y acciones
def detect_objects(img):
    results = yolo_model(img)
    return results.pandas().xyxy[0]  # DataFrame con las detecciones

# Aplicación de Streamlit
st.title("Identificación de Animales y sus Acciones")

# Entrada de URL o Base64
opcion = st.selectbox("Selecciona el tipo de entrada", ["URL", "Base64"])
if opcion == "URL":
    url = st.text_input("Ingresa la URL de la imagen de un animal", "http://images.cocodataset.org/val2017/000000039769.jpg")
    if url:
        try:
            img = Image.open(requests.get(url, stream=True).raw)
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")
else:
    base64_str = st.text_area("Ingresa la cadena en Base64 de la imagen")
    if base64_str:
        try:
            img = decodificar_imagen_base64(base64_str)
        except Exception as e:
            st.error(f"Error al decodificar la imagen: {e}")

# Clasificar la imagen y mostrar resultados
if 'img' in locals():
    st.image(img, caption="Imagen de Entrada", use_column_width=True)
    img_array = np.array(img)
    
    # Clasificar el animal
    animal_classification = classify_image(img)
    st.subheader("Clasificación del Animal")
    for cls in animal_classification:
        label_traducido = traducir_texto(cls[1])
        st.write(f"{label_traducido}: {cls[2]*100:.2f}%")
    
    # Detectar objetos y acciones
    detections = detect_objects(img)
    st.subheader("Detecciones")
    st.write(detections)
