import numpy as np
import librosa
from streamlit import cache_resource

from lib.audio_utils import load_model
from models.model_names import Models

# Define your emotion labels
emotion_labels = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

class SpeechEmotionRecognition:
    def __init__(self, model_path):
        self.path = model_path
        self.interpreter = load_model(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_emotion(self, audio_file):
        try:
            data, sampling_rate = librosa.load(audio_file, sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=0)
            x = np.expand_dims(x, axis=-1)
            self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(np.float32))
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(output_data, axis=1)[0]
            emotion = self.convert_class_to_emotion(predicted_class)
            return emotion, None  # Returning emotion and no error
        except Exception as e:
            return None, str(e)  # Returning no emotion and error message

    @staticmethod
    def convert_class_to_emotion(pred):
        return emotion_labels.get(pred, 'Unknown')

@cache_resource
def get_model():
    # Load the model here
    return SpeechEmotionRecognition(Models.ser_tflite)