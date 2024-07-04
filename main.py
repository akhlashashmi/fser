# ---- dependencies
# pip install streamlit
# pip install streamlit-option-menu
# pip install streamlit-audiorec
# pip install streamlit-player
# pip install streamlit-aggrid


import cv2
import streamlit as st
from st_pages import Page, Section, show_pages

facial_emotion_detection = "fer_live.py"
facial_emotion_detection_file = "fer_file.py"
image_emotions_recognition = "image_emotions_recognition.py"
speech_emotion_detection = "ser.py"
speech_emotion_detection_record = "record_ser.py"

# entry point of the application
def main():
    show_pages(
        [
            # Section represents the category of in the side bar
            # Page represents the page of the application
            Section("Facial Emotion Detection"),
            Page(facial_emotion_detection, "📷  Live Video"),
            Page(facial_emotion_detection_file, "🗃️  Video File"),
            Page(image_emotions_recognition, "🌌  Image File"),
            Section("Speech Emotion Detection"),
            Page(speech_emotion_detection, "🎵  Audio File"),
            Page(speech_emotion_detection_record, "🎙️  Record Audio"),
        ]
    )

if __name__ == '__main__':
    main()