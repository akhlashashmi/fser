from tempfile import NamedTemporaryFile


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


def dominent_emotion(emotions: dict, threshold: float = 0.2):
    """
    Determines the dominant emotion, considering the closeness of top emotions,
    and returns the emotion(s) along with their confidence level (percentage).

    Args:
        emotions (dict): A dictionary of emotions and their corresponding counts.
        threshold (float): The threshold for determining if emotions are close in frequency.

    Returns:
        str: The dominant emotion(s) with confidence percentages.S
    """
    # Remove the "neutral" emotion if it exists
    emotions.pop('neutral', None)

    # If no other emotions are left, return "neutral" or another placeholder emotion
    if not emotions:
        return "neutral (100%)"

    # Calculate total count for percentage calculation
    total_count = sum(emotions.values())

    # Sort emotions by count in descending order
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    # Find the top two emotions
    top_emotion, top_count = sorted_emotions[0]

    # Calculate confidence (percentage) for the top emotion
    top_confidence = (top_count / total_count) * 100

    # If there's only one emotion left, return it with confidence
    if len(sorted_emotions) == 1:
        return f"{top_emotion} ({top_confidence:.2f}%)"

    second_emotion, second_count = sorted_emotions[1]
    second_confidence = (second_count / total_count) * 100

    # Calculate the relative difference between the top two emotions
    relative_difference = abs(top_count - second_count) / top_count

    # If the emotions are close in count (within the threshold), return both with confidence
    if relative_difference <= threshold:
        return f"{top_emotion} ({top_confidence:.2f}%) | {second_emotion} ({second_confidence:.2f}%)"

    # Otherwise, return the top emotion with its confidence
    return f"{top_emotion} ({top_confidence:.2f}%)"

