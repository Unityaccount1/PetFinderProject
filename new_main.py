import streamlit as st
import numpy as np
from PIL import Image
import requests
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# Cargar el modelo y el procesador de ViLT
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Función para obtener la predicción de ViLT
def obtener_prediccion_vilt(img, pregunta):
    # Preprocesar la imagen y la pregunta
    inputs = processor(images=img, text=pregunta, return_tensors="pt")
    # Obtener la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    # Obtener la respuesta más probable
    respuesta_id = outputs.logits.argmax(-1).item()
    respuesta = processor.tokenizer.decode([respuesta_id])
    return respuesta

# Aplicación de Streamlit
st.title("Clasificación de Animales y Acciones usando ViLT")

# Entrada de URL
url = st.text_input("Ingresa la URL de la imagen de un animal", "http://images.cocodataset.org/val2017/000000039769.jpg")
pregunta = st.text_input("Ingresa una pregunta sobre la imagen", "¿Qué está haciendo el animal?")

if url:
    try:
        # Cargar la imagen desde la URL
        img = Image.open(requests.get(url, stream=True).raw)
        
        # Mostrar la imagen de entrada
        st.image(img, caption="Imagen de Entrada", use_column_width=True)
        
        # Obtener la predicción de ViLT
        respuesta = obtener_prediccion_vilt(img, pregunta)
        
        # Mostrar la predicción
        st.subheader("Predicción")
        st.write(f"Respuesta: {respuesta}")
        
    except Exception as e:
        st.error(f"Error al cargar la imagen: {e}")
