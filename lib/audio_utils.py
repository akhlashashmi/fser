from pydub import AudioSegment
from streamlit import cache_resource
from tensorflow.lite.python.interpreter import Interpreter


def process_audio_data(audio_data):
    if audio_data:
        # Create an AudioSegment object from the recorded audio data
        audio_segment = AudioSegment(audio_data, sample_width=2, frame_rate=44100, channels=1)
        audio_path = "output/latest_recording.wav"

        # Export the audio data as a WAV file
        audio_segment.export(audio_path, format="wav")
        return audio_path
    return None

@cache_resource
def load_model(model_path):
    """loads the model and cached it. this improves the performance of the app."""
    return Interpreter(model_path=model_path)