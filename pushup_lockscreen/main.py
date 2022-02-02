import numpy as np

from inference import Inference
from landmark_online import PreprocessorVideo
from preprocessing import select_landmarks
import global_config


class PushupLockscreen:
    def __init__(self):
        self.landmark_camera = PreprocessorVideo()
        self.inference_engine = Inference()

    def run(self):
        while True:
            # Run blazepose on next frame
            frame, mediapipe_landmarks = self.landmark_camera.get_next_frame()
            if frame is None:
                break

            if mediapipe_landmarks:
                # Turn the mediapipe landmarks into a numnpy array of our own format (which includes a batch dimension)
                landmarks = np.array([mediapipe_landmarks.landmarks])

                # Preprocess landmarks (selecting subset of landmarks)
                selected_landmarks = select_landmarks(landmarks, global_config.selected_keypoints)

                # Run inference on selected landmarks
                prediction = self.inference_engine.predict(selected_landmarks)

                # Print or plot prediction
                print(prediction)


if __name__ == '__main__':
    plck = PushupLockscreen()
    plck.run()
