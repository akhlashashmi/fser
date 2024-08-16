import cv2
from streamlit import title
from pages.facial_emotion_recognition import EmotionDetector


def build_ui():
    title("Facial Emotion Recognition")
    detector = EmotionDetector()
    detector.fer_live_cam(cv2.VideoCapture(0))

build_ui()
