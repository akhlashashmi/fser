from collections import Counter
import cv2
import streamlit as st
from lib.fer import FacialEmotionDetection, predicted_emotions
from lib.utils import get_uploaded_file, dominent_emotion


def process_video(file_path, emotion_placeholder) -> None:
    detector = FacialEmotionDetection()
    cap = cv2.VideoCapture(file_path)

    # Streamlit video display
    st_frame = st.empty()  # Placeholder for displaying video frames in Streamlit

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
            all_emotions = [emotion for frame in predicted_emotions for emotion in frame]
            dominant_emotion = dominent_emotion(Counter(all_emotions))
            emotion_placeholder.info(f'Dominant Emotion: {dominant_emotion}')

    cap.release()

def build_ui() -> None:
    st.title("Facial Emotion Recognition")
    file = st.file_uploader("Pick a video (MP4, MOV, MKV, AVI)", type=["mp4", "mov", "avi", "mkv"])

    emotion_placeholder = st.empty()  # Placeholder for displaying the dominant emotion

    if file is not None:
        file_path = get_uploaded_file(file)  # Save the uploaded file to a temporary location
        process_video(file_path, emotion_placeholder)  # Process the video and display the emotion-detected frames


build_ui()
