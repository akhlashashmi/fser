import streamlit as st
from pages.page_names import Pages


def main() -> None:
    # This page detects emotion from live camera
    fer_live = st.Page(Pages.FER_LIVE, title="FER Live", icon=":material/live_tv:", )
    # This page detects emotion from video file
    fer_file = st.Page(Pages.FER_FILE, title="FER File", icon=":material/video_library:", )
    # This page detects emotion from a static image
    image_fer = st.Page(Pages.IFER, title="Image FER File", icon=":material/photo:", )
    # This page detects emotion from a live audio
    ser_live = st.Page(Pages.SER_FILE, title="SER Live", icon=":material/graphic_eq:", )
    # This page detects emotion from an audio file
    ser_record = st.Page(Pages.SER_RECORDED, title="SER Recorded", icon=":material/mic:", )
    # This page detects emotion from a video file which also include audio
    fser = st.Page(Pages.FSER, title="FSER", icon=":material/music_video:", )

    # Navigation
    pg = st.navigation({
        'Facial Emotion Recognition': [fer_live, fer_file, image_fer],
        'Speech Emotion Recognition': [ser_live, ser_record],
        'Facial Speech Emotion Recognition': [fser]
    })

    st.set_page_config(page_title="Emotion Recognition", page_icon=":material/emoji_emotions:", )
    pg.run()


if __name__ == '__main__':
    main()
