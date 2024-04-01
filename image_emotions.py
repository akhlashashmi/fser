import cv2
import matplotlib.pyplot as plt
from fer import FER
from streamlit import file_uploader, image
from facial_expression_detection import EmotionDetector
from utils import get_uploaded_file

def detect_emotion(image_path):
    # Load the image
    # img = cv2.imread(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize the FER detector
    detector = FER()

    # Detect emotions in the image
    emotions = detector.detect_emotions(img)

    # Get the bounding boxes and emotions
    for emotion in emotions:
        for box, emotion_dict in emotion.items():
            emotion_name = max(emotion_dict, key=emotion_dict.get)
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with detected emotions
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# add_page_title()
# write('Live Facial Expression Detection')
# file = file_uploader("Pick a video", type=["jpeg", "jpg", "png"])

# if file is not None:
#     file_path = get_uploaded_file(file)
#     image(file_path, use_column_width=True)
detect_emotion(r'C:\Users\akhla\Downloads\multiple_emotions2.jpg')
