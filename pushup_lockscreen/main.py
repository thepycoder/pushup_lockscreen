import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib

from inference import Inference
from landmark_online import PreprocessorVideo
from preprocessing import select_landmarks
import global_config

plt.ion()


class PushupLockscreen:
    def __init__(self):
        self.landmark_camera = PreprocessorVideo()
        self.inference_engine = Inference()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.set_ylim(0, 2)
        self.line, = ax.plot(range(250), [0] * 250, 'b-')
        self.predictions = deque(maxlen=250)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

        print(matplotlib.get_backend())

    def run(self):
        while True:
            # Run blazepose on next frame
            frame, mediapipe_landmarks = self.landmark_camera.get_next_frame()
            if frame is None:
                break

            if mediapipe_landmarks:
                # Turn the mediapipe landmarks into a numpy array of our own format (which includes a batch dimension)
                landmarks = np.array([mediapipe_landmarks.landmarks])

                # Preprocess landmarks (selecting subset of landmarks)
                selected_landmarks = select_landmarks(landmarks, global_config.selected_keypoints)

                # Run inference on selected landmarks
                prediction = self.inference_engine.predict(selected_landmarks)

                # Print or plot prediction
                self.predictions.append(prediction)
                prediction_list = [x[0] + 1 for x in list(self.predictions)]
                prediction_list.extend([0] * (250 - len(prediction_list)))
                print(prediction_list)
                self.line.set_ydata(prediction_list)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()


if __name__ == '__main__':
    plck = PushupLockscreen()
    plck.run()
