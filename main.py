import streamlit as st
import numpy as np
from PIL import Image
import requests
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
import base64
from io import BytesIO

# Cargar el modelo y el procesador de ViLT solo una vez
st.write("Cargando el modelo y el procesador de ViLT...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
st.write("Modelo y procesador cargados.")

# Función para obtener la predicción de ViLT
def obtener_prediccion_vilt(img, pregunta):
    st.write("Procesando la imagen y la pregunta...")
    # Preprocesar la imagen y la pregunta
    inputs = processor(images=img, text=pregunta, return_tensors="pt")
    st.write("Imagen y pregunta procesadas.")
    
    st.write("Realizando la predicción...")
    # Obtener la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    st.write("Predicción realizada.")
    
    # Obtener la respuesta más probable
    respuesta_ids = outputs.logits.argmax(dim=-1).item()
    respuesta = processor.tokenizer.decode(respuesta_ids)
    
    # Filtrar la respuesta para asegurar que no contiene tokens no usados
    respuesta = respuesta.replace('[UNK]', '').replace('[PAD]', '').strip()
    
    return respuesta

# Función para decodificar una imagen en base64
def decodificar_imagen_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

# Aplicación de Streamlit
st.title("Clasificación de Animales y Acciones usando ViLT")

# Entrada de URL o Base64
opcion = st.selectbox("Selecciona el tipo de entrada", ["URL", "Base64"])
if opcion == "URL":
    url = st.text_input("Ingresa la URL de la imagen de un animal", "http://images.cocodataset.org/val2017/000000039769.jpg")
    if url:
        try:
            st.write("Cargando la imagen desde la URL...")
            img = Image.open(requests.get(url, stream=True).raw)
            st.write("Imagen cargada.")
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")
else:
    base64_str = st.text_area("Ingresa la cadena en Base64 de la imagen")
    if base64_str:
        try:
            st.write("Decodificando la imagen desde Base64...")
            img = decodificar_imagen_base64(base64_str)
            st.write("Imagen decodificada.")
        except Exception as e:
            st.error(f"Error al decodificar la imagen: {e}")

# Pregunta sobre la imagen
pregunta = st.text_input("Ingresa una pregunta sobre la imagen", "¿Qué está haciendo el animal?")

if 'img' in locals() and pregunta:
    try:
        # Mostrar la imagen de entrada
        st.image(img, caption="Imagen de Entrada", use_column_width=True)
        
        # Obtener la predicción de ViLT
        respuesta = obtener_prediccion_vilt(img, pregunta)
        
        # Mostrar la predicción
        st.subheader("Predicción")
        st.write(f"Respuesta: {respuesta}")
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
