from sklearn.preprocessing import StandardScaler
from clearml import Task
import numpy as np
import joblib

from training import select_landmarks
import global_config


class Inference:
    def __init__(self):
        training_task = Task.get_task(task_id='4cbfaf6975df463b89ab28379b00639b')
        self.scaler = joblib.load(training_task.artifacts['scaler_remote'].get_local_copy())
        self.model = joblib.load(training_task.artifacts['model_remote'].get_local_copy())

    def preprocess_landmarks(self, landmarks):
        self.scaler.transform(landmarks)
        return landmarks

    def run_model(self, landmarks):
        predicted = self.model.predict(landmarks)
        return predicted

    def predict(self, landmarks):
        scaled_landmarks = self.preprocess_landmarks(landmarks)
        predicted = self.run_model(scaled_landmarks)
        return predicted


if __name__ == '__main__':
    inference_engine = Inference()
    landmarks = np.array([np.loadtxt('../landmarks/pushup_up/0.csv', delimiter=',')])
    selected_landmarks = select_landmarks(landmarks, global_config.SELECTED_KEYPOINTS)
    predicted = inference_engine.predict(selected_landmarks)
    print(predicted)
