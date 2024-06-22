import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests

# Load the pre-trained EfficientNetB0 model with top layers
efficientnet_model = EfficientNetB0(weights='imagenet')

# Function to classify the image using EfficientNetB0
def classify_image(img):
    # Resize the image to 224x224 pixels as expected by EfficientNetB0
    img = img.resize((224, 224))
    # Convert the image to an array and preprocess it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Classify the image
    predictions = efficientnet_model.predict(img_array)
    return decode_predictions(predictions, top=3)[0]  # Decode top 3 predictions

# Streamlit application
st.title("Animal Image Classification using EfficientNetB0")

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

if 'img' in locals():
    
    try:
        # Load the image from the URL
        #img = Image.open(requests.get(url, stream=True).raw)
        
        # Display the input image
        st.image(img, caption="Input Image", use_column_width=True)
        
        # Classify the image and get the top 3 predictions
        predictions = classify_image(img)
        
        # Display the predictions
        st.subheader("Top 3 Predictions")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i+1}. {label}: {score:.4f}")
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
