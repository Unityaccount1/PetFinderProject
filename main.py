import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import requests
# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False)
# Function to extract features using VGG16
def extract_vgg_features(img):
    # Load and preprocess the image
    #img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Extract features
    features = vgg_model.predict(img_array)
    return features.flatten() # Flatten the feature tensor into a 1D array
def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    features = extract_vgg_features(img)
    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">Bienvenido a la página principal</h1>
            <sup style="margin-left:5px;font-size:small; color: green;">version 0.1</sup>
        </div>
        """,unsafe_allow_html=True,)
    st.write("Salida:" + features)
if __name__ == "__main__":
    main()
