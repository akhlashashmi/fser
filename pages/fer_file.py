import cv2
from streamlit import title, empty, file_uploader
from lib.utils import get_uploaded_file
from lib.fer import FacialEmotionDetection


def process_video(file_path):
    detector = FacialEmotionDetection()
    cap = cv2.VideoCapture(file_path)

    # Streamlit video display
    st_frame = empty()  # Placeholder for displaying video frames in Streamlit

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion in the current frame
        processed_frame = detector.detect_emotion_in_video_frame(frame)

        # Display the processed frame in the Streamlit app
        st_frame.image(processed_frame, channels="BGR")

    cap.release()


def build_ui() -> None:
    title("Facial Emotion Recognition")
    file = file_uploader("Pick a video (MP4, MOV, MKV, AVI)", type=["mp4", "mov", "avi", "mkv"])

    if file is not None:
        file_path = get_uploaded_file(file)  # Save the uploaded file to a temporary location
        process_video(file_path)  # Process the video and display the emotion-detected frames


build_ui()
