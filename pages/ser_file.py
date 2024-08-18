from audio_recorder_streamlit import audio_recorder
from streamlit import empty, write, success, audio, title, markdown
from lib.ser import get_model
from lib.audio_utils import process_audio_data


def build_ui() -> None:
    title("Speech Emotion Recognition (Record)")

    # Audio recorder component in Streamlit
    audio_data = audio_recorder(
        "",
        pause_threshold=2,
        neutral_color="#303030",
        recording_color="#de1212",
        icon_name="microphone",
        icon_size="8x",
    )

    # Placeholder for predicted label and audio
    info_ = empty()

    if audio_data:
        # Process the recorded audio data
        audio_path = process_audio_data(audio_data)

        # # Empty space
        # markdown('#')

        # Display a placeholder while processing
        with info_:
            write("⏳ Processing Audio...")

        # Run prediction
        model = get_model()
        predicted_label, error = model.predict_emotion(audio_path)

        # Clear the placeholder and display the result
        info_.empty()

        if error:
            error(f'Prediction failed: {error}')
        elif predicted_label:
            success(f'Predicted Emotion: {predicted_label}')
            audio(audio_path)
        else:
            error('Failed to predict emotion')

# Call build_ui at the end
build_ui()
