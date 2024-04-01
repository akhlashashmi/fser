# ---- dependencies
# pip install streamlit
# pip install streamlit-option-menu
# pip install streamlit-audiorec
# pip install streamlit-player
# pip install streamlit-aggrid


import cv2
from st_pages import Page, Section, show_pages


def main():
    show_pages(
        [
            Section("Facial Emotion Detection"),
            Page("video_emotions_live.py", "📷  Live Video"),
            Page("video_emotions_file.py", "🗃️  Video File"),
            Page("image_emotions.py", "🌌  Image File"),
            Section("Speech Emotion Detection"),
            Page("audio_emotions.py", "🎙️  Record Audio"),
            Page("audio_emotions_file.py", "🎵  Audio File"),
        ]
    )


if __name__ == '__main__':
    main()
