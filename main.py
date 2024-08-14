# ---- dependencies
# pip install streamlit
# pip install streamlit-option-menu
# pip install streamlit-audiorec
# pip install streamlit-player
# pip install streamlit-aggrid

import cv2
import streamlit as st
# from st_pages import Page, Section, show_pages

facial_emotion_detection = "fer_live.py"
facial_emotion_detection_file = "fer_file.py"
image_emotions_recognition = "image_emotions_recognition.py"
speech_emotion_detection = "ser.py"
speech_emotion_detection_record = "record_ser.py"

# entry point of the application
def main():
    # show_pages(
    #     [
    #         # Section represents the category of in the side bar
    #         # Page represents the page of the application
    #         Section("Facial Emotion Detection"),
    #         Page(facial_emotion_detection, "📷  Live Video"),
    #         Page(facial_emotion_detection_file, "🗃️  Video File"),
    #         Page(image_emotions_recognition, "🌌  Image File"),
    #         Section("Speech Emotion Detection"),
    #         Page(speech_emotion_detection, "🎵  Audio File"),
    #         Page(speech_emotion_detection_record, "🎙️  Record Audio"),
    #     ]
    # )

    fer_live = st.Page(facial_emotion_detection, title="FER Live", icon=":material/add_circle:")
    fer_file = st.Page(facial_emotion_detection_file, title="FER File", icon=":material/add_circle:")
    image_fer = st.Page(image_emotions_recognition, title="Image FER File", icon=":material/add_circle:")
    ser_live = st.Page(speech_emotion_detection, title="SER Live", icon=":material/add_circle:")
    # ser_file = st.Page(facial_emotion_detection_file, title="FER File", icon=":material/add_circle:")
    ser_record = st.Page(speech_emotion_detection_record, title="FER Recorded", icon=":material/add_circle:")

    pg = st.navigation([fer_file, fer_live, image_fer, ser_live, ser_record])
    st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
    pg.run()

if __name__ == '__main__':
    main()