import cv2
from streamlit import write, title, file_uploader
from st_pages import add_page_title

from facial_expression_detection import EmotionDetector
from utils import get_uploaded_file

# add_page_title()
# write('Live Facial Expression Detection')
title("Facial Emotion Prediction")
file = file_uploader("Pick a video", type=["mp4", "mov", "avi", "mkv"])

if file is not None:
    file_path = get_uploaded_file(file)
    # write(file_path)
    # video(file_path)
    detector = EmotionDetector()
    detector.fer_live_cam(cv2.VideoCapture(file_path))



