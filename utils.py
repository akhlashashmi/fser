from tempfile import NamedTemporaryFile
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import numpy as np

def get_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Create a temporary file
        temp_file = NamedTemporaryFile(delete=False)

        # Write the uploaded file contents to the temporary file
        temp_file.write(uploaded_file.getvalue())

        # Close the temporary file
        temp_file.close()

        # Get the file path of the temporary file
        file_path = temp_file.name

        return file_path
    else:
        return None