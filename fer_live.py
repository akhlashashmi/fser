import cv2
import streamlit as st
from facial_emotion_recognition import EmotionDetector


def live_facial_expresion_recognition():
    st.title("Facial Emotion Recognition")
    detector = EmotionDetector()
    detector.fer_live_cam(cv2.VideoCapture(0))


if __name__ == "__main__":
    live_facial_expresion_recognition()
