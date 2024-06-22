import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests

# Load the pre-trained VGG16 model with top layers
vgg_model = VGG16(weights='imagenet')

# Function to classify the image using VGG16
def classify_image(img):
    # Resize the image to 224x224 pixels as expected by VGG16
    img = img.resize((224, 224))
    # Convert the image to an array and preprocess it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Classify the image
    predictions = vgg_model.predict(img_array)
    return decode_predictions(predictions, top=3)[0]  # Decode top 3 predictions

# Streamlit application
st.title("Animal Image Classification using VGG16")

# URL input
url = st.text_input("Enter image URL of an animal", "http://images.cocodataset.org/val2017/000000039769.jpg")

if url:
    try:
        # Load the image from the URL
        img = Image.open(requests.get(url, stream=True).raw)
        
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
