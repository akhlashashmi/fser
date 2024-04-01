import streamlit as st
from audio_recorder_streamlit import audio_recorder
from audio_utils import *
from pydub import AudioSegment

col1, col2, col3 = st.columns(3)

with col1:
    st.text('')
with col2:
    audio_data = audio_recorder('',
    pause_threshold= 3,
    neutral_color= "#303030",
    recording_color = "#de1212",
    icon_name = "microphone",
    icon_size = "10x",)
with col3:
    st.text('')


if audio_data:
    # st.audio(audio_data, format="audio/wav")
    audio_segment = AudioSegment(audio_data, sample_width=2, frame_rate=44100, channels=2)
    audio_segment.export("current_recording.wav", format="wav")

    model = get_model('models/speech_sentiment_analysis.h5')
    audio_path = 'current_recording.wav'

    predicted_emotion = predict_emotion(audio_path, model)
    predicted_label = emotion_labels[predicted_emotion]

    st.success(f"Predicted Emotion: {predicted_label}")
    st.audio(audio_path)  # Play the uploaded audio

