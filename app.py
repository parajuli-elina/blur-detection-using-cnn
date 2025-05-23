import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Function to predict if the image is blur or focus
def predict_image(image):
    # Load the pre-trained model
    model_path=r'.\blur-detection-using-cnn\cnn_model\model90'
    cnn_model = tf.keras.models.load_model(model_path)

    # Define class names
    class_names = ('blur', 'focus')
    img_size=(128, 128, 3)

    image = np.array(image)
    resized_image = cv2.resize(image, img_size[0:2])[:, :, ::-1]  # Resize to the input size of the model
    #resized_image = resized_image / 255.0  # Normalize the image
    prediction = cnn_model.predict(resized_image[None, ...], verbose=0)[0]
    return class_names[np.argmax(prediction)]

# Streamlit app
st.title("Blur Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a spinner while predicting
    with st.spinner('Predicting...'):
        # Predict if the image is blur or focus
        prediction = predict_image(image)
    
    st.write(f"Prediction: {prediction}")