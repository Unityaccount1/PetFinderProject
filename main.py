import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from transformers import MarianMTModel, MarianTokenizer
import base64
from io import BytesIO

# Cargar el modelo preentrenado EfficientNetB0 con las capas superiores
efficientnet_model = EfficientNetB0(weights='imagenet')

# Cargar el modelo y el tokenizador para la traducción
translation_model_name = 'Helsinki-NLP/opus-mt-en-es'
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

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
    inputs = translation_tokenizer(texto, return_tensors="pt", truncation=True)
    translated_tokens = translation_model.generate(**inputs)
    traduccion = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return traduccion[0]

# Función para decodificar una imagen en base64
def decodificar_imagen_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

# Aplicación de Streamlit
st.title("Clasificación de Imágenes de Animales usando EfficientNetB0")

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
    predictions = classify_image(img)
    st.subheader("Las 3 Principales Predicciones")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        label_traducido = traducir_texto(label)
        st.write(f"{i+1}. {label_traducido}: {score:.4f}")
