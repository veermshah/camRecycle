import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import random

MODELSPATH = 'waste_model.h5'

# Function to load models
@st.cache
def load_models():
    model = load_model(MODELSPATH, compile=False)
    return model

# Set page title and background (code continues...)
# Set page title and background
st.set_page_config(
    page_title="Recycling Importance & Waste Prediction",
    page_icon="🔄",
    layout="wide",
)
# Sidebar for page selection (code continues...)
# Sidebar for page selection
st.sidebar.title('Select options:')
page = st.sidebar.selectbox("Choose a page:", ["Sample Data", "Upload an Image", "Matching"])

# Initialize variable to hold the captured image
captured_image = None

if page == "Take a photo":
    st.title("Take a photo for waste prediction")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    if st.button("Take Photo"):
        st.text("I am taking the photo...")
        saved_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        saved_frame = Image.fromarray(saved_frame)
        st.image(saved_frame, channels="RGB", use_column_width=True)

        saved_frame_path = "captured_frame.png"
        saved_frame.save(saved_frame_path, format="PNG")

        cap.release()

        # Assign the captured image to the global variable
        captured_image = saved_frame

        # Add processing message
        st.text("I am processing your image...")

# Provide an option to upload the captured image in the "Upload an Image" page
if page == "Upload an Image":
    st.header("Your prediction of waste")

    if captured_image is not None:
        st.info("Show your image:")
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        img = captured_image.resize((64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of your image'):
            model = load_models()
            result = model.predict(img)
            st.write(result)
            if result[0][0] == 1:
                prediction = "It is recyclable waste and you are good to recycle it"
            elif result[0][0] == 0:
                prediction = "It is the organic waste you cannot recycle, But it is good if you decompose it in the soil."
            else:
                prediction = "The model could not make a confident prediction."
            st.write(prediction)
            st.success("Success!")