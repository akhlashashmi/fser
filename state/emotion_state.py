class EmotionState:
    def __init__(self):
        self.state = {}

    def update(self, new_state):
        self.state = new_state

    def get(self):
        return self.state

emotion_state = EmotionState()
