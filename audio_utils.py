import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "surprise",
    6: "sad",
}


@st.cache_resource
def get_model(model_path):
    """loads the model and cached it. this improves the performance of the app."""
    return  tf.keras.models.load_model(model_path)


def extract_mfcc(audio_path, duration=3, offset=0.5, n_mfcc=40):
    """loads the model and cached it. this improves the performance of the app."""
    y, sr = librosa.load(audio_path, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc.reshape(1, -1, 1)


def predict_emotion(audio_path, model):
    '''prediction'''
    mfccs = extract_mfcc(audio_path)
    prediction = model.predict(mfccs)
    predicted_emotion = np.argmax(prediction)
    return predicted_emotion
