import cv2
from streamlit import title, file_uploader
from pages.facial_emotion_recognition import EmotionDetector
from utils import get_uploaded_file


def build_ui():
    title("Facial Emotion Recognition")
    file = file_uploader(
        "Pick a video (MP4, MOV, MKV, AVI)", type=["mp4", "mov", "avi", "mkv"]
    )
    if file is not None:
        file_path = get_uploaded_file(file)
        detector = EmotionDetector()
        detector.fer_live_cam(cv2.VideoCapture(file_path))

build_ui()
