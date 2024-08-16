import streamlit as st

from lib.ser import get_model
from lib.utils import get_uploaded_file


def build_ui():
    st.title("Speech Emotion Recognition (Upload)")

    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        audio_path = get_uploaded_file(uploaded_file)

        model = get_model()

        predicted_label, error = model.predict_emotion(audio_path)

        if error:
            st.error(f'Prediction failed: {error}')
        elif predicted_label:
            st.success(f"Predicted Emotion: {predicted_label}")
            st.audio(audio_path)
        else:
            st.error('Failed to predict emotion')


# Call build_ui at the end
build_ui()