import cv2
from streamlit import write, audio
from st_pages import add_page_title
from facial_expression_detection import EmotionDetector


# add_page_title()
# write('Live Facial Expression Detection')
detector = EmotionDetector()
detector.fer_live_cam(cv2.VideoCapture(0))

