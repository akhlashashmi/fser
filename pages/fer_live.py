import cv2
import streamlit as st
from lib.fer import FacialEmotionDetection


def build_ui() -> None:
    st.title("Facial Emotion Recognition - Live Camera")
    detector = FacialEmotionDetection()

    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)

    # If the camera is successfully opened
    if video_capture.isOpened():
        st_frame = st.empty()  # A placeholder for video frames
        while True:
            ret, frame = video_capture.read()  # Read a frame from the camera
            if not ret:
                break

            # Detect emotions in the frame
            frame_with_emotion = detector.detect_emotion_in_video_frame(frame)

            # Display the frame with detected emotions
            st_frame.image(frame_with_emotion, channels="BGR")

            # If the user presses "q", stop the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        st.error("Unable to access the camera.")

    video_capture.release()
    cv2.destroyAllWindows()


build_ui()
