import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import requests

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features using VGG16
def extract_vgg_features(img):
    # Load and preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Extract features
    features = vgg_model.predict(img_array)
    return features.flatten() # Flatten the feature tensor into a 1D array

# Streamlit application
st.title("VGG16 Feature Extraction")

# URL input
url = st.text_input("Enter image URL", "http://images.cocodataset.org/val2017/000000039769.jpg")

if url:
    try:
        img = Image.open(requests.get(url, stream=True).raw)
        st.image(img, caption="Input Image", use_column_width=True)
        features = extract_vgg_features(img)
        st.write("Extracted Features:")
        st.write(features)
    except Exception as e:
        st.error(f"Error loading image: {e}")
