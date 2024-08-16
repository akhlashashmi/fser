import streamlit as st
from audio_utils import *
from models.model_names import Models
from utils import get_uploaded_file


def build_ui():
    st.title("Speech Emotion Recognition")

    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV or MP3)", type=["wav", "mp3"]
    )

    if uploaded_file is not None:
        audio_path = get_uploaded_file(uploaded_file)
        model = get_model(Models.ser)

        predicted_emotion = predict_emotion(audio_path, model)
        predicted_label = emotion_labels[predicted_emotion]

        st.success(f"Predicted Emotion: {predicted_label}")
        st.audio(audio_path)  # Play the uploaded audio


build_ui()
