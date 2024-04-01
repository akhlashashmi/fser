import streamlit as st
from audio_utils import *
from utils import get_uploaded_file


def main():
    """Streamlit app to predict emotion from audio"""

    st.title("Audio Emotion Prediction")

    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        audio_path = get_uploaded_file(uploaded_file)
        model = get_model('models/speech_sentiment_analysis.h5')
        
        predicted_emotion = predict_emotion(audio_path, model)
        predicted_label = emotion_labels[predicted_emotion]

        st.success(f"Predicted Emotion: {predicted_label}")
        st.audio(audio_path)  # Play the uploaded audio

if __name__ == "__main__":
    main()



# from streamlit import write, audio, button, columns, text
# from st_pages import add_page_title
# from streamlit import write, file_uploader
# from keras.models import load_model
# from utils import get_uploaded_file
# import numpy as np
# import librosa

# model = load_model('models/speech_sentiment_analysis.h5')

# # Function to extract MFCCs from an audio file
# def extract_mfcc(filename):
#     y, sr = librosa.load(filename, duration=3, offset=0.5)
#     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#     return mfcc.reshape(1, -1, 1)  # Reshape for model input

# # Function to predict emotion from an audio file
# def predict_emotion(filename):
    
#     mfccs = extract_mfcc(filename)
#     prediction = model.predict(mfccs)  # Get model prediction
#     predicted_emotion = np.argmax(prediction)  # Convert one-hot encoded prediction to emotion index
#     return predicted_emotion

# from IPython.display import Audio

# file = file_uploader("Pick a video", type=["wav", "mp3"])


# if file is not None:
#     audio_file_path = get_uploaded_file(file)

#     # Predict emotion for the given audio file
#     predicted_emotion = predict_emotion(audio_file_path)

#     # You'll probably have a dictionary or mapping for integer indices to emotion labels
#     emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'surprise', 6: 'sad'}

#     # Print the predicted emotion
#     print("Predicted emotion:", emotion_labels[predicted_emotion])
#     audio(audio_file_path)
#     write(f'Detected Emotion: {emotion_labels[predicted_emotion]}')


# References
# https://medium.com/@soukaina./building-a-speech-emotion-analyzer-in-python-5a8d4ac332fc
# https://www.youtube.com/watch?v=-VQL8ynOdVg


