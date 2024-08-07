import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2

# TensorFlow Model Prediction
def model_prediction(image_array, confidence_threshold=0.5):
    model = tf.keras.models.load_model("trained_model.h5")
    input_arr = np.array(image_array).astype(np.float32) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)  # convert single image to batch
    predictions = model.predict(input_arr)
    max_index = np.argmax(predictions)
    max_confidence = predictions[0][max_index]
    
    if max_confidence < confidence_threshold:
        return None
    return max_index

# Set Page Config
st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* General CSS */
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #6c757d;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #5a6268;
    }
    .stFileUploader {
        border: 2px dashed #6c757d;
        border-radius: 10px;
    }
    /* Header CSS */
    .css-18e3th9 {
        background-color: #6c757d;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 28px;
        text-align: center;
    }
    /* Sidebar CSS */
    .css-1d391kg {
        background-color: #6c757d;
        border-radius: 10px;
        padding: 20px;
    }
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    .css-1d391kg .css-1d391kg {
        background-color: #5a6268;
    }
    /* Container CSS */
    .css-1e5imcs {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.image("sidebar_logo.png", use_column_width=True)
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.write("## Navigation")
st.sidebar.write("Use the sidebar to navigate through the app.")

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # Camera Processor
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.result_index = None
            self.class_names = [
                'Apple___Apple_scab_hallo', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            self.model = tf.keras.models.load_model("trained_model.h5")

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
            except Exception as e:
                st.error(f"Error converting frame: {e}")
                return frame
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (128, 128))
            result_index = model_prediction(img_resized)

            if result_index is not None:
                cv2.putText(
                    img, f"{self.class_names[result_index]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.write("Or upload an image file for prediction")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            test_image = st.file_uploader("Choose an Image:")
            if st.button("Show Image") and test_image:
                st.image(test_image, use_column_width=True)

        with col2:
            if st.button("Predict"):
                if test_image:
                    with st.spinner("Please wait..."):
                        img = Image.open(test_image)
                        result_index = model_prediction(np.array(img.resize((128, 128))))
                        # Reading Labels
                        class_name = ['Apple___Apple_scab_hallo', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                      'Tomato___healthy']
                        if result_index is None:
                            st.error("The disease could not be recognized with confidence.")
                        else:
                            st.success(f"Model is predicting it's a {class_name[result_index]}")
                else:
                    st.error("Please upload an image first.")
