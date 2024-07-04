import streamlit as st
from audio_recorder_streamlit import audio_recorder
from audio_utils import *
from pydub import AudioSegment

def process_audio(audio_data):
    if audio_data:
        audio_segment = AudioSegment(audio_data, sample_width=2, frame_rate=44100, channels=2)
        audio_path = "current_recording.wav"
        audio_segment.export(audio_path, format="wav")

        model = get_model('models/speech_sentiment_analysis.h5')
        predicted_emotion = predict_emotion(audio_path, model)
        predicted_label = emotion_labels[predicted_emotion]

        return predicted_label, audio_path
    return None, None

def main():
    st.title("Speech Emotion Recognition")

    # Giving Some Horizontal Spacing
    st.markdown("#")

    # _, col2, _ = st.columns(3)

    # with col2:
    #     audio_data = audio_recorder(
    #         "",
    #         pause_threshold=3,
    #         neutral_color="#303030",
    #         recording_color="#de1212",
    #         icon_name="microphone",
    #         icon_size="10x",
    #     )

    audio_data = audio_recorder(
            "",
            pause_threshold=3,
            neutral_color="#303030",
            recording_color="#de1212",
            icon_name="microphone",
            icon_size="10x",
        )

    space = st.empty()

    predicted_label, audio_path = process_audio(audio_data)

    if predicted_label is not None and audio_path is not None:
        space = st.markdown("#")
        st.success(f"Predicted Emotion: {predicted_label}")
        st.audio(audio_path)  # Play the uploaded audio

if __name__ == "__main__":
    main()
