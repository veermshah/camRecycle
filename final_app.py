import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import random
import cv2

MODELSPATH = 'waste_model.h5'
DATAPATH = 'waste1.jpg'
IMAGE_DIR = 'images' # Directory containing recycling images


# Function to load data
@st.cache
def load_data():
    img = Image.open(DATAPATH)
    return img

# Function to load models
def load_models():
    model = load_model(MODELSPATH, compile=False)
    return model

# Set page title and background
st.set_page_config(
    page_title="Recycling Importance & Waste Prediction",
    page_icon="🔄",
    layout="wide",
)

# Add images related to recycling
recycling_images = "recycle.png"
st.image(recycling_images, caption=[None], width=300)

st.title("Recyclability Image Classifier")

# Add a title with a background image
st.markdown(
    """
    <style>
        .title-text {
            font-size: 48px;
            font-weight: bold;
            color: #000000;
            text-shadow: 2px 2px 4px #000000;
        }
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            background-image: url('https://www.arlingtontx.gov/UserFiles/Servers/Server_14481062/Image/City%20Hall/Government/Environmental_Commitment/Library_Drop-Off_Recycling.jpg');
            background-size: cover;
            background-position: center;
            height: 300px;
            padding: 30px;
            border-radius: 10px;
        }
    </style>
    
    """,
    unsafe_allow_html=True
)

# Add some information about recycling
st.markdown(
    """
    Recycling is crucial for preserving our environment and conserving natural resources. It helps reduce pollution, 
    saves energy, and reduces the need for raw materials. By recycling, we can contribute to a more sustainable and 
    eco-friendly future.
    """
)

# Sidebar for page selection
st.sidebar.title('Select options:')
page = st.sidebar.selectbox("Choose a page:", ["Sample Data", "Upload an Image", "Take a Photo", "Matching"])

if page == "Sample Data":
    st.header("Sample prediction for waste")
    if st.checkbox('Show Sample Image'):
        st.info("Sample Image:")
        image = load_data()
        st.image(image, caption='Sample Image', use_column_width=True)
        image = image.resize((64, 64))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of Sample Image'):
            model = load_models()
            result = model.predict(image)
            st.write(result)
            if result[0][0] == 1:
                prediction = 'Recyclable Waste'
            else:
                prediction = 'Organic Waste, you cannot recycle this, but you can decompose it in soil.'
            st.write(prediction)
            st.success("Successful")

if page == "Upload an Image":
    st.header("Upload an Image")
    
    #UPLOAD
    uploaded_file = st.file_uploader("Choose your image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.info("Show your image:")
        st.image(img, caption="Upload image", use_column_width=True)
        img = img.resize((64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of your image'):
            model = load_models()
            result = model.predict(img)
            st.write(result)
            if result[0][0] == 1:
                prediction = 'Recyclable Waste'
            else:
                prediction = 'Organic Waste, you cannot recycle this, but you can decompose it in soil.'
            st.write(prediction)
            st.success("Success!")
    
if page == "Take a Photo":
    # Check for the "Take Photo" button click
    if st.button("Take Photo"):
        st.text("I am taking the photo...")  # Added line to indicate the photo is being taken
        
        # Access the camera feed using OpenCV
        cap = cv2.VideoCapture(0)

        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Convert the captured frame to a format suitable for saving
        #saved_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        cv2.imwrite("captured_photo.jpg", frame)
        
        img = Image.open("captured_photo.jpg")
        st.info("Show your image:")
        st.image(img, caption="Upload image", use_column_width=True)
        img = img.resize((64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        model = load_models()
        result = model.predict(img)
        st.write(result)
        if result[0][0] == 1:
            prediction = 'Recyclable Waste'
        else:
            prediction = 'Organic Waste, you cannot recycle this, but you can decompose it in soil.'
        st.write(prediction)
        st.success("Success!")
    
if page == "Matching":
    st.title("Webcam Photo")
    # Access the camera feed using OpenCV
    cap = cv2.VideoCapture(0)

    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to a format compatible with PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Display the frame in Streamlit
    #st.image(frame, channels="RGB", use_column_width=True)

    # Check for the "Take Photo" button click
    if st.button("Take Photo"):
        st.text("I am taking the photo...")  # Added line to indicate the photo is being taken
        
        # Convert the captured frame to a format suitable for saving
        saved_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        saved_frame = Image.fromarray(saved_frame)
        st.image(saved_frame, channels="RGB", use_column_width=True)
    
        x = random.random()
        if x < 0.3:
            image_path = "jakey.png"
            st.image(image_path, caption='Jake<3', use_column_width=True)
            st.success("Jake<3")
        else:
            image_path = "nrup.png"
            st.image(image_path, caption='Nrup<3', use_column_width=True)
            st.success("Nrup<3")
