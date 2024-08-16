import streamlit as st
from PIL import Image
from lib.fer import FacialEmotionDetection


def build_ui() -> None:
    st.title("Facial Emotion Recognition - Image Upload")

    # File uploader for images
    image_file = st.file_uploader("Choose an image file (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        image = Image.open(image_file)

        # Detect emotion in the uploaded image
        detector = FacialEmotionDetection()
        detector.detect_emotion_in_image(image)


build_ui()
