import cv2
import streamlit as st
import plotly.express as px
from collections import Counter
from lib.fer import FacialEmotionDetection, predicted_emotions
from lib.utils import get_uploaded_file, dominent_emotion
from state.emotion_state import emotion_state


def plot_emotion_distribution(chart_placeholder, emotion_counts: dict) -> None:
    # Calculate total emotions for percentage calculation
    total = sum(emotion_counts.values())
    if total > 0:
        percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}

        # Create a bar chart using Plotly
        fig = px.bar(
                x=list(percentages.keys()),
                y=list(percentages.values()),
                labels={'x': 'Emotion', 'y': 'Percentage'},
                title="Emotion Distribution",
            )

        # Update the chart in the placeholder
        chart_placeholder.plotly_chart(fig, use_container_width=True)


def process_video(file_path) -> None:
    detector = FacialEmotionDetection()
    cap = cv2.VideoCapture(file_path)

    # Streamlit video display
    st_frame = st.empty()  # Placeholder for displaying video frames in Streamlit

    # Placeholder for displaying the dominant emotion
    emotion_placeholder = st.empty()
    # Placeholder for displaying the emotion distribution chart
    chart_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion in the current frame
        processed_frame = detector.detect_emotion_in_video_frame(frame)

        # Display the processed frame in the Streamlit app
        st_frame.image(processed_frame, channels="BGR")

        # Update the dominant emotion in real-time
        if len(predicted_emotions) > 0:
            emotion_dict = emotion_state.get()
            dominant_emotion = dominent_emotion(emotion_dict)
            emotion_placeholder.info(f'{dominant_emotion}')

            # Update the emotion distribution graph without clearing the placeholder
            plot_emotion_distribution(chart_placeholder, emotion_dict)

    cap.release()


def build_ui() -> None:
    st.title("Facial Emotion Recognition")
    file = st.file_uploader("Pick a video (MP4, MOV, MKV, AVI)", type=["mp4", "mov", "avi", "mkv"])

    if file is not None:
        # Save the uploaded file to a temporary location
        file_path = get_uploaded_file(file)
        # Process the video and display the emotion-detected frames
        process_video(file_path)


build_ui()
