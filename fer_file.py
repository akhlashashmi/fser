import cv2
import streamlit as st

# from st_pages import add_page_title
from facial_emotion_recognition import EmotionDetector
from utils import get_uploaded_file


def facial_expresion_recognition():
    st.title("Facial Emotion Recognition")
    file = st.file_uploader(
        "Pick a video (MP4, MOV, MKV, AVI)", type=["mp4", "mov", "avi", "mkv"]
    )
    if file is not None:
        file_path = get_uploaded_file(file)
        detector = EmotionDetector()
        detector.fer_live_cam(cv2.VideoCapture(file_path))


# if __name__ == "__main__":
facial_expresion_recognition()
